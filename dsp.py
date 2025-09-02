from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class TraceParams:
    run_ms: float = 1000.0        
    ep_min_ms: float = 200.0      
    ep_max_ms: float = 700.0
    ar_coeff: float = 0.8         
    base_mean: float = 10.0
    drift_per_ms: float = 0.002
    base_noise_std: float = 0.5
    step_mean: float = 2.0
    step_jitter_std: float = 0.5
    post_var_bump_std: float = 0.3

def simulate_trace_and_endpoint(tp: TraceParams, rng_seed: int = 11) -> Tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(rng_seed)
    n = int(tp.run_ms)  
    t = np.arange(n, dtype=float)

    
    w = np.random.randn(n) * tp.base_noise_std
    for i in range(1, n):
        w[i] = tp.ar_coeff * w[i-1] + w[i]

    base = tp.base_mean + tp.drift_per_ms * t + w
    ep_ms = float(np.random.uniform(tp.ep_min_ms, tp.ep_max_ms))
    ep_idx = int(ep_ms)

    x = base.copy()
    step_mag = tp.step_mean + tp.step_jitter_std * np.random.randn()
    x[ep_idx:] += step_mag
    x[ep_idx:] += tp.post_var_bump_std * np.random.randn(n - ep_idx)

    return t, x, ep_ms

@dataclass
class DetectorParams:
    pre_win_ms: int = 20
    post_win_ms: int = 20
    frame_ms: int = 10
    hop_ms: int = 5
    thresh_std: float = 1.5
    hyst_ms: int = 10

def detect_endpoint(
    t_ms: np.ndarray,
    x: np.ndarray,
    dp: DetectorParams
) -> Optional[float]:
    n = len(x)
    pre = int(dp.pre_win_ms)
    post = int(dp.post_win_ms)
    frame = int(dp.frame_ms)
    hop = int(dp.hop_ms)
    hyst = int(dp.hyst_ms)

    last_trigger_ms = None
    triggered = False

    for start in range(0, n - frame, hop):
        end = start + frame
        pre_start = max(0, start - pre)
        pre_end = start
        if pre_end - pre_start < max(5, pre//2):
            continue
        post_start = start
        post_end = min(n, start + post)

        pre_mean = float(np.mean(x[pre_start:pre_end]))
        pre_std  = float(np.std(x[pre_start:pre_end]) + 1e-6)
        post_mean = float(np.mean(x[post_start:post_end]))
        z = (post_mean - pre_mean) / pre_std
        now_ms = float(start)

        if z >= dp.thresh_std:
            if not triggered:
                triggered = True
                last_trigger_ms = now_ms
            if triggered and now_ms - last_trigger_ms >= hyst:
                return now_ms
        else:
            triggered = False
            last_trigger_ms = None

    return None
