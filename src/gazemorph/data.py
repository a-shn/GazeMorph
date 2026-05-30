from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


SLICE_MSS = 400
T_SIZE = 200


def normalize_times(info: dict) -> dict:
    new_start = 1000
    items = list(info["fixations"].items())
    fixations = {new_start: items[0][1]}
    last_time = int(items[0][0].split(".")[0])
    current_time = new_start
    for f_time, fixation in info["fixations"].items():
        time = int(f_time.split(".")[0])
        current_time += min(1000, abs(time - last_time))
        last_time = time
        fixations[current_time] = fixation
    return fixations


def normalize_wordbounds(info: dict) -> dict:
    wordbounds = {}
    for key, value in info["wordbounds"].items():
        wordbounds[key] = {
            "left": value["left"] / 2,
            "top": value["top"] / 2,
            "right": value["right"] / 2,
            "bottom": value["bottom"] / 2,
            "is_read": 1,
        }
    return wordbounds


def stamp_gaussian(canvas: torch.Tensor, y: int, x: int, sigma: float = 1.5, amp: float = 1.0) -> None:
    height, width = canvas.shape[-2], canvas.shape[-1]
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    radius = int(math.ceil(3 * sigma))
    y0, y1 = max(0, y - radius), min(height - 1, y + radius)
    x0, x1 = max(0, x - radius), min(width - 1, x + radius)
    yy = torch.arange(y0, y1 + 1, dtype=torch.float32) - y
    xx = torch.arange(x0, x1 + 1, dtype=torch.float32) - x
    kernel = torch.exp(-(yy[:, None].pow(2) + xx[None, :].pow(2)) / (2.0 * sigma * sigma))
    patch = canvas[y0 : y1 + 1, x0 : x1 + 1].to(torch.float32)
    patch += amp * kernel
    canvas[y0 : y1 + 1, x0 : x1 + 1] = patch.to(canvas.dtype)


def downsample_mask_preserve_gaps(mask_2d: torch.Tensor) -> torch.Tensor:
    x = mask_2d.unsqueeze(0).unsqueeze(0)
    y = 1.0 - F.max_pool2d(1.0 - x, kernel_size=2, stride=2)
    return y.squeeze(0).squeeze(0)


def load_sample(path: str | Path, use_content: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    path = Path(path)
    info = json.loads((path / "info.json").read_text())
    wordbounds_tensor = torch.load(path / "wordbounds_tensor.pt").to(torch.float32)
    content = downsample_mask_preserve_gaps(wordbounds_tensor)
    if not use_content:
        content = torch.ones_like(content)
    content = content.to(torch.float16)
    fixations = normalize_times(info)
    wordbounds = normalize_wordbounds(info)
    moving_pts = torch.tensor(
        [[int(time / SLICE_MSS), int(value["x"]) // 2, int(value["y"]) // 2] for time, value in fixations.items()],
        dtype=torch.float16,
    )
    fixed_pts_list = []
    for f_time, fixation in fixations.items():
        x = fixation.get("x_corrected", fixation.get("x_before"))
        y = fixation.get("y_corrected", fixation.get("y_before"))
        fixed_pts_list.append([int(f_time / SLICE_MSS), int(math.floor(x)) // 2, int(math.floor(y)) // 2])
    fixed_pts = torch.tensor(fixed_pts_list, dtype=torch.float16)
    height, width = content.shape[-2], content.shape[-1]
    frames = [torch.zeros((height, width), dtype=torch.float32) for _ in range(T_SIZE)]
    for time, fixation in fixations.items():
        t_idx = int(time / SLICE_MSS)
        if 0 <= t_idx < T_SIZE:
            amp = min(1.0, max(0.0, float(fixation["duration"]) / 300.0))
            stamp_gaussian(frames[t_idx], int(fixation["y"]) // 2, int(fixation["x"]) // 2, amp=amp)
    fixations_3d = torch.stack([frame.to(torch.float16) for frame in frames], dim=0)
    content_3d = content.repeat(T_SIZE, 1, 1)
    page_3d = torch.stack([fixations_3d, content_3d], dim=0)
    return page_3d, moving_pts, fixed_pts, wordbounds


class AugmentedZucoTrainDataset(Dataset):
    def __init__(self, root_dir: str | Path, use_content: bool = True):
        self.root_dir = Path(root_dir)
        self.use_content = use_content
        self.ids = sorted((alpha.name, int(item.name)) for alpha in self.root_dir.iterdir() for item in alpha.iterdir())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        alpha, item_id = self.ids[idx]
        return load_sample(self.root_dir / alpha / str(item_id), use_content=self.use_content)


class AugmentedZucoTestDataset(Dataset):
    def __init__(self, root_dir: str | Path, use_content: bool = True):
        self.root_dir = Path(root_dir)
        self.use_content = use_content
        self.ids = sorted(int(path.name) for path in self.root_dir.iterdir())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return load_sample(self.root_dir / str(self.ids[idx]), use_content=self.use_content)


def collate_as_lists(batch):
    pages, moving_pts, fixed_pts, wordbounds = zip(*batch)
    return torch.stack(pages, dim=0), list(moving_pts), list(fixed_pts), list(wordbounds)
