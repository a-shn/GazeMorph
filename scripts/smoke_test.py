from __future__ import annotations

import argparse

import torch

from gazemorph.checkpoint import load_model
from gazemorph.utils import default_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=str(default_device()))
    parser.add_argument("--local-dir", default="artifacts/models")
    parser.add_argument("--shape", nargs=3, type=int, default=(24, 24, 24))
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_model(args.model, checkpoint_path=args.checkpoint, device=device, local_dir=args.local_dir)
    t, h, w = args.shape
    x = torch.zeros((1, 2, t, h, w), dtype=torch.float32, device=device)
    with torch.inference_mode():
        y = model(x)
    print(f"model:  {args.model}")
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
