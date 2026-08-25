"""ImageNet-1K linear-probe classifier builders for the demo — ConvMAE-Base vs Ghost+ConvMAE.

Wraps the research model factories (bundled under ``linprobe/``) into one loader that mirrors
``face_models.build_and_load`` but for the *frozen-backbone + linear-probe head* classifiers
(the 90-epoch linear probe trained on top of the 300-epoch pretrained backbones).

Only the two convolution/Transformer arms are here — the same two the rest of this demo ships. The
Ghost+ForwardMamba / Ghost+BiMamba arms are deliberately **not** included (see README): their
Stage-3 Mamba blocks need the CUDA-only ``mamba_ssm`` selective-scan kernels, which do not install
on the target laptop GPU, and they have no ONNX/TensorRT path — so they cannot run in this demo.

The linprobe head is ``Sequential(BatchNorm1d(768, affine=False), Linear(768, 1000))``; a checkpoint
is detected as a probe by its ``head.1.*`` keys and the head is rebuilt to match before loading.
"""
import os
import sys

import torch
import torch.nn as nn

# The bundled factories use flat top-level imports of each other, so put their directory on
# sys.path and import flat (same pattern as face_models.py with models/).
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linprobe")
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from model_convmae_cls_baseline import convmae_baseline_cls  # noqa: E402
from model_convmae_cls_ghost import convmae_ghost_cls        # noqa: E402

CLS_ARMS = ("baseline", "ghost")
_BUILDERS = {"baseline": convmae_baseline_cls, "ghost": convmae_ghost_cls}


def build_and_load_cls(arm, num_classes, ckpt_path, map_location="cpu"):
    """Build the arm's linprobe classifier and load a checkpoint. Asserts a clean load."""
    model = _BUILDERS[arm](num_classes=num_classes)
    sd = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    # linprobe checkpoints wrap the head as Sequential(BatchNorm1d, Linear)
    if any(k.startswith("head.1.") for k in sd):
        model.head = nn.Sequential(
            nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing]
    assert not missing, f"{arm}: missing keys on load: {missing[:8]}"
    model.eval()
    return model
