import csv
import matplotlib.pyplot as plt

def bar_from_csv(csv_path: str, title: str, metric_keys, out_png: str):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return
    labels = [row["config"] for row in rows]
    for key in metric_keys:
        plt.figure()
        vals = [float(row[key]) for row in rows]
        x = range(len(vals))
        plt.bar(x, vals)
        plt.xticks(list(x), labels, rotation=20)
        plt.ylabel(key)
        plt.title(f"{title} — {key}")
        plt.tight_layout()
        plt.savefig(out_png.replace(".png", f"_{key}.png"), dpi=160)
