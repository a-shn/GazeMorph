# GazeMorph

Code for GazeMorph, a 3D UNet-style model for fixation-to-content alignment in eye-tracking data.

## Install

```bash
pip install -e .
```

## Pretrained Models

| name | base channels | Hugging Face |
| --- | ---: | --- |
| small | 16 | `qskxx/gazemorph_small` |
| medium | 32 | `qskxx/gazemorph_medium` |
| large | 48 | `qskxx/gazemorph_large` |

## Dataset

The augmented ZuCo dataset archive is hosted at:

```text
qskxx/gazemorph_zuco_augmented
```

## Download Artifacts

```bash
python scripts/download_artifacts.py
```

## Smoke Test

```bash
python scripts/smoke_test.py --model small
```

With a local checkpoint:

```bash
python scripts/smoke_test.py --model small --checkpoint path/to/gmorph_small.pt
```

## Python Usage

```python
import torch
from gazemorph import load_model

model = load_model("small")
x = torch.zeros(1, 2, 24, 24, 24)
with torch.inference_mode():
    flow = model(x)
```

## Notebooks

Development notebooks are in `notebooks/`.
