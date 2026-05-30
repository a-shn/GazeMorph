from __future__ import annotations

import torch
from torch.nn import functional as F


def spatial_smoothness(flow: torch.Tensor) -> torch.Tensor:
    dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    return 0.5 * (dx.square().mean() + dy.square().mean())


def temporal_smoothness(flow: torch.Tensor) -> torch.Tensor:
    dt = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    return dt.square().mean()


def jacobian_fold_penalty(flow: torch.Tensor) -> torch.Tensor:
    u = flow[:, 0:1]
    v = flow[:, 1:2]
    ux = u[:, :, :, :, 1:] - u[:, :, :, :, :-1]
    uy = u[:, :, :, 1:, :] - u[:, :, :, :-1, :]
    vx = v[:, :, :, :, 1:] - v[:, :, :, :, :-1]
    vy = v[:, :, :, 1:, :] - v[:, :, :, :-1, :]
    h = min(ux.shape[-2], uy.shape[-2], vx.shape[-2], vy.shape[-2])
    w = min(ux.shape[-1], uy.shape[-1], vx.shape[-1], vy.shape[-1])
    ux = ux[..., :h, :w]
    uy = uy[..., :h, :w]
    vx = vx[..., :h, :w]
    vy = vy[..., :h, :w]
    det = (1 + ux) * (1 + vy) - uy * vx
    return F.relu(-det).mean()


def point_correspondence_loss(flow: torch.Tensor, moving_pts_list: list[torch.Tensor], fixed_pts_list: list[torch.Tensor]) -> torch.Tensor:
    batch, _, time, height, width = flow.shape
    total = flow.new_tensor(0.0)
    count = 0
    for b in range(batch):
        moving = moving_pts_list[b]
        fixed = fixed_pts_list[b]
        if moving.numel() == 0 or fixed.numel() == 0:
            continue
        length = min(moving.shape[0], fixed.shape[0])
        moving = moving[:length].to(flow.device, dtype=torch.float32)
        fixed = fixed[:length].to(flow.device, dtype=torch.float32)
        t = moving[:, 0].long().clamp_(0, time - 1)
        x = moving[:, 1].long().clamp_(0, width - 1)
        y = moving[:, 2].long().clamp_(0, height - 1)
        dx = flow[b, 0, t, y, x]
        dy = flow[b, 1, t, y, x]
        total += F.smooth_l1_loss(moving[:, 1] + dx, fixed[:, 1], reduction="mean")
        total += F.smooth_l1_loss(moving[:, 2] + dy, fixed[:, 2], reduction="mean")
        count += 1
    return total if count == 0 else total / count


def gazemorph_loss(
    flow: torch.Tensor,
    moving_pts_list: list[torch.Tensor],
    fixed_pts_list: list[torch.Tensor],
    lambda_spatial: float = 0.035,
    lambda_time: float = 0.05,
    lambda_fold: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    point = point_correspondence_loss(flow, moving_pts_list, fixed_pts_list)
    spatial = spatial_smoothness(flow)
    temporal = temporal_smoothness(flow)
    fold = jacobian_fold_penalty(flow)
    total = point + lambda_spatial * spatial + lambda_time * temporal + lambda_fold * fold
    parts = {"point": point, "spatial": spatial, "temporal": temporal, "fold": fold}
    return total, parts
