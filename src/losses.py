#!/usr/bin/env python3
"""
OCEANSR -- masked loss for stage-1 training.

The model outputs a full (B,1,T,H,W) skin-SST field, but supervision exists only
on the held-out day's clear-water pixels, per sensor, after the model's learned
per-sensor offset. This module:

  * gathers the prediction at each sample's target day (target_pos varies in the
    batch),
  * applies `model.sensor_offset(sensor)` (0 for the anchor sensor),
  * computes a masked Huber loss over the true observed pixels for ECOSTRESS and
    Landsat, averaged over the sensors that actually have pixels,
  * (optional) a small total-variation term on the predicted day.

All quantities are in the shared, z-scored SST space.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from data import CHANNELS

_CLOUD_IDX = CHANNELS.index("cloud_cover")   # HRRR total cloud cover channel (0..1)


def masked_huber(pred, target, mask, delta: float = 1.0):
    """Huber over pixels where mask>0.5. Returns (loss, n_valid)."""
    m = mask > 0.5
    n = int(m.sum())
    if n == 0:
        return pred.sum() * 0.0, 0          # keep graph, zero contribution
    return F.huber_loss(pred[m], target[m], delta=delta, reduction="mean"), n


def sst_masked_loss(pred, batch, model, delta: float = 1.0, tv_weight: float = 0.0,
                    cloud_max: float | None = None):
    """pred: (B,1,T,H,W). batch carries per-sensor target/mask (B,H,W) + target_pos.

    cloud_max (0..1, fraction): if set, supervision is dropped at target-day pixels
    whose HRRR total cloud cover exceeds it -- so the loss only sees clear-sky days."""
    B = pred.shape[0]
    dev = pred.device
    tp = batch["target_pos"].to(dev).long()
    # prediction at each sample's masked day -> (B,H,W)
    pred_day = pred[torch.arange(B, device=dev), 0, tp]

    # cloud gate from the (unblanked) HRRR cloud-cover channel at the target day
    cloud_ok = None
    if cloud_max is not None and "x" in batch:
        xb = batch["x"]                                   # may be on CPU (not moved to GPU)
        cloud_day = xb[torch.arange(B, device=xb.device), _CLOUD_IDX,
                       tp.to(xb.device)].to(dev)
        cloud_ok = (cloud_day <= cloud_max).float()      # (B,H,W) 1 where clear enough

    total = pred.sum() * 0.0
    metrics = {}
    n_sensors = 0
    for s, tkey, mkey in (("eco", "eco_target", "eco_mask"),
                          ("lst", "lst_target", "lst_mask")):
        tgt = batch[tkey].to(dev)
        msk = batch[mkey].to(dev)
        if cloud_ok is not None:
            msk = msk * cloud_ok                          # keep only clear-sky supervision
        b = model.sensor_offset(s)                 # 0.0 (anchor) or Parameter(1,)
        adj = pred_day + b
        loss_s, n = masked_huber(adj, tgt, msk, delta)
        if n > 0:
            total = total + loss_s
            n_sensors += 1
            with torch.no_grad():
                sel = msk > 0.5
                rmse = torch.sqrt(((adj - tgt)[sel] ** 2).mean())
            metrics[f"{s}_rmse"] = float(rmse)
            metrics[f"{s}_px"] = n
    if n_sensors > 0:
        total = total / n_sensors

    if tv_weight > 0:
        tv = ((pred_day[:, :, 1:] - pred_day[:, :, :-1]).abs().mean()
              + (pred_day[:, 1:, :] - pred_day[:, :-1, :]).abs().mean())
        total = total + tv_weight * tv
        metrics["tv"] = float(tv.detach())

    metrics["loss"] = float(total.detach())
    metrics["n_sensors"] = n_sensors
    return total, metrics


if __name__ == "__main__":  # tiny smoke test (needs torch)
    import sys
    sys.path.insert(0, "src")
    from model import UNet3D
    from data import num_input_channels, MUR_INDEX
    B, T, H, W = 2, 8, 32, 32
    m = UNet3D(in_channels=num_input_channels(), depth=3, mur_index=MUR_INDEX)
    pred = m(torch.randn(B, num_input_channels(), T, H, W))
    batch = {
        "target_pos": torch.tensor([T - 1, T // 2]),
        "eco_target": torch.randn(B, H, W), "eco_mask": (torch.rand(B, H, W) > 0.7).float(),
        "lst_target": torch.randn(B, H, W), "lst_mask": (torch.rand(B, H, W) > 0.9).float(),
    }
    loss, mets = sst_masked_loss(pred, batch, m, delta=1.0)
    loss.backward()
    print("loss", float(loss), "metrics", mets)
    print("lst offset grad:", m.sensor_offsets["lst"].grad)
    print("OK")