#!/usr/bin/env python3
"""
OCEANSR -- stage-1 training (3D U-Net, masked whole-day reconstruction).

    python src/train.py --config configs/config.yaml

Builds the model + train/val loaders, trains with AdamW + warmup-cosine LR and
BF16 autocast, validates each epoch (held-out AOIs), and checkpoints the best
model. Metrics are appended to <ckpt_dir>/metrics.jsonl. RMSE is reported both in
normalized units and Kelvin (× the shared SST std).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import UNet3D                       # noqa: E402
from data import make_loader, num_input_channels, MUR_INDEX  # noqa: E402
from losses import sst_masked_loss             # noqa: E402


def make_scheduler(opt, warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    acc = {"eco": [0.0, 0], "lst": [0.0, 0]}
    for batch in loader:
        pred = model(batch["x"].to(device))
        B = pred.shape[0]
        tp = batch["target_pos"].to(device).long()
        pd = pred[torch.arange(B, device=device), 0, tp]
        for s, tk, mk in (("eco", "eco_target", "eco_mask"), ("lst", "lst_target", "lst_mask")):
            tgt = batch[tk].to(device); msk = batch[mk].to(device) > 0.5
            adj = pd + model.sensor_offset(s)
            acc[s][0] += float(((adj - tgt) ** 2)[msk].sum())
            acc[s][1] += int(msk.sum())
    out = {}
    for s, (se, n) in acc.items():
        out[f"{s}_rmse"] = math.sqrt(se / n) if n else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="override train.steps_per_epoch")
    ap.add_argument("--val-steps", type=int, default=None, help="override train.val_steps")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None, help="0 = single-process (best for debugging)")
    ap.add_argument("--log-every", type=int, default=None, help="log a step point every N steps (default 50)")
    ap.add_argument("--cpu", action="store_true", help="force CPU (smoke test without GPU)")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    # CLI overrides for quick smoke tests
    ov = cfg.setdefault("train", {})
    for key, val in (("steps_per_epoch", args.steps), ("val_steps", args.val_steps),
                     ("batch_size", args.batch_size), ("num_workers", args.num_workers)):
        if val is not None:
            ov[key] = val
    mcfg, tcfg = cfg.get("model", {}), cfg.get("train", {})

    torch.manual_seed(int(tcfg.get("seed", 0)))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs or int(tcfg.get("epochs", 100))
    ckpt_dir = Path(cfg.get("project", {}).get("root", ".")) / tcfg.get("ckpt_dir", "results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # data
    train_loader = make_loader(cfg, "train", seed=int(tcfg.get("seed", 0)))
    val_loader = make_loader(cfg, "val", seed=0)
    sst_std = float(train_loader.dataset.sst_std)

    # model
    model = UNet3D(
        in_channels=num_input_channels(),
        base_width=int(mcfg.get("base_width", 48)),
        depth=int(mcfg.get("depth", 3)),
        mur_index=MUR_INDEX,
        use_checkpoint=bool(mcfg.get("grad_checkpoint", True)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device} | params={n_params:.1f}M | in_ch={num_input_channels()} | sst_std={sst_std:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 3e-4)),
                            weight_decay=float(tcfg.get("weight_decay", 0.01)))
    steps_per_epoch = max(1, len(train_loader))
    sched = make_scheduler(opt, int(tcfg.get("warmup_steps", 500)), epochs * steps_per_epoch)

    amp = str(tcfg.get("amp", "bf16"))
    use_amp = amp in ("bf16", "fp16") and device == "cuda"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and device == "cuda"))
    delta = float(tcfg.get("huber_delta", 1.0))
    tv = float(tcfg.get("tv_weight", 0.0))
    clip = float(tcfg.get("grad_clip", 1.0))
    log_every = max(1, args.log_every or int(tcfg.get("log_every", 50)))

    start_epoch, best = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", float("inf"))
        print(f"resumed from {args.resume} @ epoch {start_epoch}")

    log = open(ckpt_dir / "metrics.jsonl", "a")
    steps_log = open(ckpt_dir / "steps.jsonl", "a")
    gstep = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, batch in enumerate(train_loader):
            gstep += 1
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(batch["x"].to(device))
                loss, mets = sst_masked_loss(pred, batch, model, delta=delta, tv_weight=tv)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            sched.step()
            run += mets["loss"]
            if i % log_every == 0:
                lr = sched.get_last_lr()[0]
                print(f"  e{epoch} step {i}/{steps_per_epoch} loss={mets['loss']:.4f} lr={lr:.2e}")
                steps_log.write(json.dumps(
                    {"epoch": epoch, "step": i, "gstep": gstep,
                     "loss": mets["loss"], "lr": lr}) + "\n")
                steps_log.flush()

        val = validate(model, val_loader, device)
        rec = {"epoch": epoch, "train_loss": run / steps_per_epoch,
               "val_eco_rmse_K": val["eco_rmse"] * sst_std,
               "val_lst_rmse_K": val["lst_rmse"] * sst_std,
               "sec": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"epoch {epoch}: train_loss={rec['train_loss']:.4f} "
              f"val_eco={rec['val_eco_rmse_K']:.3f}K val_lst={rec['val_lst_rmse_K']:.3f}K "
              f"({rec['sec']}s)")

        score = val["eco_rmse"] if not math.isnan(val["eco_rmse"]) else val["lst_rmse"]
        ck = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
              "best": best, "cfg": cfg, "sst_std": sst_std,
              "sst_mean": float(train_loader.dataset.sst_mean)}
        torch.save(ck, ckpt_dir / "last.pt")
        if score < best:
            best = score; ck["best"] = best
            torch.save(ck, ckpt_dir / "best.pt")
            print(f"  ** new best (val rmse {score*sst_std:.3f} K) -> best.pt")
    log.close(); steps_log.close()
    print("training done.")


if __name__ == "__main__":
    main()