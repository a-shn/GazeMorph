from __future__ import annotations

from pathlib import Path

import torch

from .model import GazeMorph


MODEL_SPECS = {
    "small": {
        "repo_id": "qskxx/gazemorph_small",
        "filename": "gmorph_small.pt",
        "base": 16,
        "max_disp": 24.0,
    },
    "medium": {
        "repo_id": "qskxx/gazemorph_medium",
        "filename": "gmorph_medium.pt",
        "base": 32,
        "max_disp": 24.0,
    },
    "large": {
        "repo_id": "qskxx/gazemorph_large",
        "filename": "gmorph_large.pt",
        "base": 48,
        "max_disp": 24.0,
    },
}


def get_model_spec(name: str) -> dict:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        names = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model '{name}'. Available models: {names}") from exc


def download_checkpoint(name: str, local_dir: str | Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    spec = get_model_spec(name)
    path = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        local_dir=str(local_dir) if local_dir is not None else None,
    )
    return Path(path)


def read_state_dict(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_model(
    name: str = "small",
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    local_dir: str | Path | None = None,
) -> GazeMorph:
    spec = get_model_spec(name)
    path = Path(checkpoint_path) if checkpoint_path is not None else download_checkpoint(name, local_dir=local_dir)
    model = GazeMorph(in_ch=2, base=spec["base"], max_disp=spec["max_disp"]).to(device)
    model.load_state_dict(read_state_dict(path, map_location=device))
    model.eval()
    return model
