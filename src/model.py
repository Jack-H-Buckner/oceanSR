#!/usr/bin/env python3
"""
OCEANSR -- 3D U-Net for spatiotemporal SST gap-filling (stage-1 benchmark).

Maps a (B, C_in, T, H, W) space-time window to a (B, 1, T, H, W) skin-SST field,
one value per day so the masked day (interior=interpolation, last=nowcasting) can
be read out. Architecture (see the design diagram):

    stem DoubleConv ─┐ skip0
       Down ─────────┐ skip1
          Down ──────┐ skip2
             Down  (bottleneck)
             Up + skip2
          Up + skip1
       Up + skip0
    head Conv3d->1  ->  residual ;  pred = MUR + residual

Key choices:
  * Pooling is SPATIAL-ONLY (1,2,2) -> the temporal axis T is preserved end to end.
  * Output is a residual added to the MUR backbone channel (always present), so the
    net only learns high-frequency detail. The MUR channel index is `mur_index`.
  * Learnable per-sensor offsets reconcile ECOSTRESS/Landsat to one output field;
    they are applied in the LOSS (see losses.py via `sensor_offset`). One sensor is
    anchored to 0 for identifiability.

NOTE for data.py: the SST-like channels (mur_sst, eco_sst, lst_sst) must share a
common SST normalization so that `MUR + residual` and the per-sensor targets live
in the same space; the per-sensor offset then absorbs the calibration difference.
Default channel order assumed here:
    0 eco_sst | 1 eco_mask | 2 lst_sst | 3 lst_mask | 4 mur_sst | 5.. forcing/static/time
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _groups(channels: int, target: int = 8) -> int:
    """Largest group count <= target that divides `channels` (for GroupNorm)."""
    g = min(target, channels)
    while channels % g:
        g -= 1
    return g


class DoubleConv(nn.Module):
    """[Conv3d(k3) -> GroupNorm -> SiLU] x 2."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(c_out), c_out),
            nn.SiLU(inplace=True),
            nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(c_out), c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Spatial-only max-pool (keep T), then DoubleConv."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = DoubleConv(c_in, c_out)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Trilinear upsample (spatial-only), concat the encoder skip, then DoubleConv."""

    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        self.conv = DoubleConv(c_in + c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:          # robust to odd sizes (e.g. full-AOI)
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, base_width: int = 48,
                 depth: int = 3, mur_index: int = 4, residual: bool = True,
                 sensors=("eco", "lst"), anchor_sensor: str = "eco",
                 use_checkpoint: bool = False):
        super().__init__()
        self.mur_index = mur_index
        self.residual = residual
        self.anchor_sensor = anchor_sensor
        self.use_checkpoint = use_checkpoint

        widths = [base_width * (2 ** i) for i in range(depth + 1)]   # e.g. [48,96,192,384]
        self.stem = DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList([Down(widths[i], widths[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList(
            [Up(widths[i + 1], widths[i], widths[i]) for i in reversed(range(depth))])
        self.head = nn.Conv3d(widths[0], out_channels, kernel_size=1)

        # per-sensor offsets; the anchor sensor is fixed at 0 (returns float 0.0)
        self.sensor_offsets = nn.ParameterDict(
            {s: nn.Parameter(torch.zeros(1)) for s in sensors if s != anchor_sensor})

    def sensor_offset(self, name: str):
        """Learnable additive offset for a sensor (0.0 for the anchor sensor)."""
        if name == self.anchor_sensor:
            return 0.0
        return self.sensor_offsets[name]

    def _maybe_ckpt(self, module, *inputs):
        if self.use_checkpoint and self.training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, x):
        mur = x[:, self.mur_index:self.mur_index + 1]      # (B,1,T,H,W), normalized backbone

        h = self.stem(x)
        skips = [h]
        for down in self.downs:
            h = self._maybe_ckpt(down, h)
            skips.append(h)

        h = skips[-1]                                      # bottleneck
        for i, up in enumerate(self.ups):
            skip = skips[-2 - i]
            h = self._maybe_ckpt(up, h, skip)

        out = self.head(h)
        return out + mur if self.residual else out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Shape test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    B, C, T, H, W = 2, 17, 16, 128, 128
    model = UNet3D(in_channels=C, base_width=48, depth=3, mur_index=4)
    x = torch.randn(B, C, T, H, W)
    with torch.no_grad():
        y = model(x)
    print(f"input  {tuple(x.shape)}")
    print(f"output {tuple(y.shape)}   (expect ({B}, 1, {T}, {H}, {W}))")
    assert y.shape == (B, 1, T, H, W), "output shape mismatch"
    print(f"params: {count_params(model)/1e6:.2f} M")
    print(f"eco offset (anchor): {model.sensor_offset('eco')}")
    print(f"lst offset (learned): {model.sensor_offset('lst').shape}")
    print("OK")