import random
from dataclasses import dataclass,field
from typing import List, Tuple, Optional, Dict
import numpy as np
import math
import os


DEFAULT_SEED = 20250901

DEFAULT_BE_LOGN_MU = 1.2
DEFAULT_BE_LOGN_SIGMA = 0.8
DEFAULT_TDMA_QUEUE_MEAN_MS = 0.2
DEFAULT_TDMA_QUEUE_STD_MS  = 0.10

DEFAULT_AIR_MEAN_MS = 3.0
DEFAULT_AIR_STD_MS  = 0.8

DEFAULT_GE_P_GB = 0.02       # P(G->B)
DEFAULT_GE_P_BG = 0.15       # P(B->G)
DEFAULT_GE_LOSS_GOOD = 0.005
DEFAULT_GE_LOSS_BAD  = 0.12


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(xs, q))

@dataclass
class ProducerParams:
    rate_hz: float = 20.0
    payload_bytes: int = 64

class Producer:
    def __init__(self, p: ProducerParams):
        self.p = p

    def emit_times_ms(self, total_time_s: float) -> List[float]:
        period_ms = 1000.0 / self.p.rate_hz
        t = 0.0
        out = []
        while t < total_time_s * 1000.0:
            out.append(t)
            t += period_ms
        return out


@dataclass
class SchedulerParams:
    mode: str = "best_effort"   
    slot_ms: float = 1.0
    chunks_per_slot: int = 1

class Scheduler:
    def __init__(self, p: SchedulerParams):
        self.p = p

    def schedule_chunk_times(self, emit_ms: float, num_chunks: int) -> List[float]:
        if self.p.mode == "best_effort":
            return [emit_ms for _ in range(num_chunks)]
        
        slot = self.p.slot_ms
        first_slot = int(np.ceil(emit_ms / slot))
        times, used_in_slot, slot_i = [], 0, first_slot
        for _ in range(num_chunks):
            if used_in_slot >= self.p.chunks_per_slot:
                slot_i += 1
                used_in_slot = 0
            times.append(slot_i * slot)
            used_in_slot += 1
        return times


@dataclass
class IIDParams:
    be_logn_mu: float = DEFAULT_BE_LOGN_MU
    be_logn_sigma: float = DEFAULT_BE_LOGN_SIGMA
    air_mean_ms: float = DEFAULT_AIR_MEAN_MS
    air_std_ms: float = DEFAULT_AIR_STD_MS
    loss_prob: float = 0.005

class IIDChannel:
    """Independent per-chunk queue jitter + air jitter + Bernoulli loss."""
    def __init__(self, p: IIDParams, seed: int = DEFAULT_SEED):
        self.p = p
        self.rng = random.Random(seed)
        np.random.seed(seed % (2**32-1))

    def queue_delay_ms(self, tdma: bool) -> float:
        if tdma:
            
            return max(0.0, self.rng.gauss(DEFAULT_TDMA_QUEUE_MEAN_MS, DEFAULT_TDMA_QUEUE_STD_MS))
        else:
            return float(np.random.lognormal(self.p.be_logn_mu, self.p.be_logn_sigma))

    def airtime_ms(self) -> float:
        return max(0.0, self.rng.gauss(self.p.air_mean_ms, self.p.air_std_ms))

    def lost(self) -> bool:
        return (self.rng.random() < self.p.loss_prob)


@dataclass
class GEParams:
    p_gb: float = DEFAULT_GE_P_GB
    p_bg: float = DEFAULT_GE_P_BG
    loss_good: float = DEFAULT_GE_LOSS_GOOD
    loss_bad: float = DEFAULT_GE_LOSS_BAD

@dataclass
class GEChannelParams:
    be_logn_mu: float = DEFAULT_BE_LOGN_MU
    be_logn_sigma: float = DEFAULT_BE_LOGN_SIGMA
    air_mean_ms: float = DEFAULT_AIR_MEAN_MS
    air_std_ms: float = DEFAULT_AIR_STD_MS
    ge: GEParams = field(default_factory=GEParams)  


class GEChannel:
    """Gilbert–Elliott bursty channel with queue jitter + air jitter."""
    def __init__(self, p: GEChannelParams, seed: int = DEFAULT_SEED):
        self.p = p
        self.rng = random.Random(seed)
        self.state_bad = False
        np.random.seed((seed + 1) % (2**32-1))

    def step_state(self):
        if self.state_bad:
            if self.rng.random() < self.p.ge.p_bg:
                self.state_bad = False
        else:
            if self.rng.random() < self.p.ge.p_gb:
                self.state_bad = True

    def queue_delay_ms(self, tdma: bool) -> float:
        if tdma:
            return max(0.0, self.rng.gauss(DEFAULT_TDMA_QUEUE_MEAN_MS, DEFAULT_TDMA_QUEUE_STD_MS))
        else:
            return float(np.random.lognormal(self.p.be_logn_mu, self.p.be_logn_sigma))

    def airtime_ms(self) -> float:
        return max(0.0, self.rng.gauss(self.p.air_mean_ms, self.p.air_std_ms))

    def lost(self) -> bool:
        loss_p = (self.p.ge.loss_bad if self.state_bad else self.p.ge.loss_good)
        return (self.rng.random() < loss_p)


@dataclass
class FECParams:
    n: int = 1
    k: int = 1
    payload_bytes: int = 64
    paths: int = 1       

class FECEncoder:
    """Idealized k-of-n erasure coding accounting."""
    def __init__(self, p: FECParams):
        assert 1 <= p.k <= p.n
        assert p.paths in (1, 2)
        self.p = p

    @property
    def overhead(self) -> float:
        return self.p.n / self.p.k


def transmit_one(
    emit_ms: float,
    fec: FECEncoder,
    sched: Scheduler,
    ch1,
    ch2=None,
    deadline_ms: float = 20.0,
) -> Tuple[bool, float, float, float]:
    """
    Returns:
      success_before_deadline (bool),
      latency_ms_if_success_or_inf,
      bytes_sent (≈ payload*n/k),
      excess_lateness_ms (0 if on-time; positive if k-th arrival after deadline; deadline penalty if never reconstructs)
    """
    times = sched.schedule_chunk_times(emit_ms, fec.p.n)
    tdma_flag = (sched.p.mode == "tdma")
    arrivals = []

    for i, st in enumerate(times):
        
        ch = ch1 if (fec.p.paths == 1 or ch2 is None or i % 2 == 0) else ch2
       
        if isinstance(ch, GEChannel):
            ch.step_state()
        
        q = ch.queue_delay_ms(tdma=tdma_flag)
        a = ch.airtime_ms()
        if ch.lost():
            continue
        arrivals.append(st + q + a)

    bytes_sent = fec.overhead * fec.p.payload_bytes

    if len(arrivals) >= fec.p.k:
        arrivals.sort()
        t_k = arrivals[fec.p.k - 1]
        lat = t_k - emit_ms
        if lat <= deadline_ms:
            return True, lat, bytes_sent, 0.0
        else:
            return False, float("inf"), bytes_sent, (lat - deadline_ms)
    else:
        return False, float("inf"), bytes_sent, deadline_ms


@dataclass
class SimParams:
    total_time_s: float = 60.0
    deadline_ms: float = 20.0
    channel_mode: str = "ge"   # "iid" or "ge"

def run_config(
    sim: SimParams,
    prod: Producer,
    sched: Scheduler,
    fec: FECEncoder,
    iid_params: Optional[IIDParams] = None,
    ge_params: Optional[GEChannelParams] = None,
    seed: int = DEFAULT_SEED
) -> Dict[str, float]:
    emits = prod.emit_times_ms(sim.total_time_s)
    N = len(emits)

    
    if sim.channel_mode == "iid":
        ch1 = IIDChannel(iid_params or IIDParams(), seed + 1)
        ch2 = IIDChannel(iid_params or IIDParams(), seed + 2) if fec.p.paths == 2 else None
    else:
        ch1 = GEChannel(ge_params or GEChannelParams(), seed + 1)
        ch2 = GEChannel(ge_params or GEChannelParams(), seed + 2) if fec.p.paths == 2 else None

    succ = 0
    latencies = []
    total_bytes = 0.0
    total_payload = N * fec.p.payload_bytes
    excess_sum = 0.0

    for t in emits:
        ok, lat, sent, excess = transmit_one(t, fec, sched, ch1, ch2, sim.deadline_ms)
        total_bytes += sent
        excess_sum += excess
        if ok:
            succ += 1
            latencies.append(lat)

    success_rate = succ / max(1, N)
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    return {
        "messages": N,
        "deadline_ms": sim.deadline_ms,
        "deadline_success": success_rate,
        "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
        "overhead": total_bytes / max(1, total_payload),
        "late_rate": 1.0 - success_rate,
        "excess_lateness_mean_ms": excess_sum / max(1, N),
    }
