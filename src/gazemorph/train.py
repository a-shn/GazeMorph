from __future__ import annotations

from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import collate_as_lists
from .losses import gazemorph_loss


def train_epoch(
    model: torch.nn.Module,
    dataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 4,
    amp_dtype: torch.dtype | None = None,
    scaler: GradScaler | None = None,
    lambda_spatial: float = 0.035,
    lambda_time: float = 0.05,
    lambda_fold: float = 0.10,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_as_lists,
    )
    model.train()
    scaler = scaler or GradScaler("cuda", enabled=False)
    sums = {"loss": 0.0, "point": 0.0, "spatial": 0.0, "temporal": 0.0, "fold": 0.0}
    enabled = device.type == "cuda" and amp_dtype is not None
    for pages, moving, fixed, _ in tqdm(loader, leave=False):
        pages = pages.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=enabled, dtype=amp_dtype or torch.float16):
            flow = model(pages)
            loss, parts = gazemorph_loss(flow, moving, fixed, lambda_spatial, lambda_time, lambda_fold)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        sums["loss"] += float(loss.detach().cpu())
        for key, value in parts.items():
            sums[key] += float(value.detach().cpu())
    return {key: value / max(1, len(loader)) for key, value in sums.items()}


def validation_loss(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 4,
    amp_dtype: torch.dtype | None = None,
    lambda_spatial: float = 0.035,
    lambda_time: float = 0.05,
    lambda_fold: float = 0.10,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_as_lists,
    )
    model.eval()
    sums = {"loss": 0.0, "point": 0.0, "spatial": 0.0, "temporal": 0.0, "fold": 0.0}
    enabled = device.type == "cuda" and amp_dtype is not None
    with torch.no_grad():
        for pages, moving, fixed, _ in tqdm(loader, leave=False):
            pages = pages.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
            with autocast("cuda", enabled=enabled, dtype=amp_dtype or torch.float16):
                flow = model(pages)
                loss, parts = gazemorph_loss(flow, moving, fixed, lambda_spatial, lambda_time, lambda_fold)
            sums["loss"] += float(loss.detach().cpu())
            for key, value in parts.items():
                sums[key] += float(value.detach().cpu())
    return {key: value / max(1, len(loader)) for key, value in sums.items()}


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    epoch: int | None = None,
    best_val: float | None = None,
) -> None:
    checkpoint = {"model": model.state_dict()}
    if optimizer is not None:
        checkpoint["optim"] = optimizer.state_dict()
    if scaler is not None and scaler.is_enabled():
        checkpoint["scaler"] = scaler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if best_val is not None:
        checkpoint["best_val"] = best_val
    torch.save(checkpoint, path)
