from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from gazemorph.checkpoint import MODEL_SPECS, download_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", choices=["small", "medium", "large"], default=["small", "medium", "large"])
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    root = Path(args.output_dir)
    models_dir = root / "models"
    data_dir = root / "data"
    for name in args.models:
        path = download_checkpoint(name, local_dir=models_dir)
        print(f"{name}: {path}")
    if not args.skip_dataset:
        path = hf_hub_download(
            repo_id="qskxx/gazemorph_zuco_augmented",
            repo_type="dataset",
            filename="data_different_alphas.zip",
            local_dir=str(data_dir),
        )
        print(f"dataset: {path}")


if __name__ == "__main__":
    main()
