from __future__ import annotations

import torch


def warp_points_3d(
    pts_txy: torch.Tensor,
    warp_x: torch.Tensor,
    warp_y: torch.Tensor,
    clamp_to_bounds: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if warp_x.ndim == 3:
        warp_x = warp_x.unsqueeze(0)
        warp_y = warp_y.unsqueeze(0)
    if warp_x.ndim != 4 or warp_y.ndim != 4:
        raise ValueError("warp_x and warp_y must have shape (T, H, W) or (B, T, H, W)")
    _, time, height, width = warp_x.shape
    pts = pts_txy.to(device=warp_x.device, dtype=torch.float32)
    t = pts[:, 0].to(torch.long)
    x = pts[:, 1].to(torch.long)
    y = pts[:, 2].to(torch.long)
    if clamp_to_bounds:
        t = t.clamp(0, time - 1)
        x = x.clamp(0, width - 1)
        y = y.clamp(0, height - 1)
    dx = warp_x[0, t, y, x]
    dy = warp_y[0, t, y, x]
    warped = torch.stack([pts[:, 0], pts[:, 1] + dx, pts[:, 2] + dy], dim=1)
    disp = torch.stack([dx, dy], dim=1)
    return warped, disp


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
