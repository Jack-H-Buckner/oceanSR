#!/usr/bin/env python3
"""
Plot OCEANSR training progress from the logs train.py writes.

    python src/plot_progress.py                       # one-off snapshot
    python src/plot_progress.py --watch 30            # refresh every 30 s

Reads <ckpt_dir>/steps.jsonl (per-step train loss) and metrics.jsonl (per-epoch
val RMSE in Kelvin) and writes results/training_progress.png. The --watch mode
re-reads and redraws on an interval so you can monitor a running job (the PNG
updates in place; open it in a viewer that auto-reloads, or re-open it).
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path):
    rows = []
    if Path(path).exists():
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def draw(ckpt_dir, out):
    steps = read_jsonl(Path(ckpt_dir) / "steps.jsonl")
    epochs = read_jsonl(Path(ckpt_dir) / "metrics.jsonl")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    if steps:
        gs = [s["gstep"] for s in steps]
        ax[0].plot(gs, [s["loss"] for s in steps], lw=1.2, color="#1f77b4")
        ax[0].set_xlabel("step"); ax[0].set_ylabel("train loss (Huber)")
        ax[0].set_title(f"train loss  (step {gs[-1]})")
        ax[0].grid(alpha=0.3)
    else:
        ax[0].text(0.5, 0.5, "no steps.jsonl yet", ha="center", va="center")

    if epochs:
        e = [r["epoch"] for r in epochs]
        ax[1].plot(e, [r.get("val_eco_rmse_K") for r in epochs], "-o", ms=3, label="ECOSTRESS")
        ax[1].plot(e, [r.get("val_lst_rmse_K") for r in epochs], "-o", ms=3, label="Landsat")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val RMSE (K)")
        best = min((r.get("val_eco_rmse_K", float("inf")) for r in epochs), default=float("nan"))
        ax[1].set_title(f"val RMSE  (best eco {best:.3f} K)")
        ax[1].grid(alpha=0.3); ax[1].legend()
    else:
        ax[1].text(0.5, 0.5, "no epochs logged yet", ha="center", va="center")

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return len(steps), len(epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="results/checkpoints")
    ap.add_argument("--out", default="results/training_progress.png")
    ap.add_argument("--watch", type=float, default=0.0, help="refresh interval (s); 0 = once")
    args = ap.parse_args()

    while True:
        ns, ne = draw(args.ckpt_dir, args.out)
        print(f"wrote {args.out}  ({ns} step-points, {ne} epochs)")
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()