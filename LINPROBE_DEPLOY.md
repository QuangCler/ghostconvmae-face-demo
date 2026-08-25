# Deploying the 90-epoch linear-probe classifiers

This guide covers the **ImageNet-1K linear-probe (linprobe) classifiers** — the 90-epoch linear
heads trained on top of the **frozen 300-epoch MAE-pretrained backbones**, one per arm. It is the
representation-quality deployment that sits alongside the face fine-tune demo in this repo (the face
tabs deploy *fine-tuned* models; this deploys the *frozen-backbone + linear-head* probes).

Linear probing freezes the whole encoder and trains **only** a linear head on ImageNet-1K, so it
measures the linear separability of the pretrained features directly — the discriminating metric in
the report (Figure VI-2).

## Deployment matrix — read this first

| Arm | Stage-3 block | Reference linprobe Top-1 | **Deploy pipelines** |
|---|---|---|---|
| **baseline** (ConvMAE-Base) | Transformer × 11 | 64.06 % | **PyTorch + TensorRT** |
| **ghost** (Ghost + ConvMAE) | Transformer × 11 | 58.60 % | **PyTorch + TensorRT** |
| **bimamba** (Ghost + BiMamba) | Bi-Mamba-2 × 7 + Transformer × 4 `{3,7,9,10}` | 58.60 %¹ | **PyTorch only** |
| **forwardmamba** (Ghost + ForwardMamba) | Mamba-2 × 7 + Transformer × 4 `{3,7,9,10}` | 57.10 %¹ | **PyTorch only** |

**The two convolution/Transformer arms (baseline, ghost) run through both native PyTorch and a full
TensorRT engine. The two state-space arms (bimamba, forwardmamba) run through native PyTorch only.**
The Mamba selective-scan operation has no standard ONNX operator, so the ONNX → TensorRT export path
has no clean engine to build — a toolchain limitation, not an architectural one (mirrors the report's
mixed-pipeline rationale and Table VI-4b, where TensorRT rows exist only for base/ghost). The
deployment tool enforces this automatically: `--pipeline auto` adds TensorRT for base/ghost and stays
on PyTorch for the Mamba arms; `--pipeline tensorrt` on a Mamba arm prints the "not supported by the
tested export path" message and falls back to PyTorch.

> ¹ **base/ghost** Top-1 are the W&B `convmae-linprobe` runs
> (`convmae_base_lineprobe_resume90_from_ep39`, `allghost_ep300_linprobe_resume90_keep_lr`; ghost
> logs 58.57, rounded to 58.60). **bimamba/forwardmamba** Top-1 are the 90-epoch runs whose
> checkpoints you deploy below (`outputs/{bimamba,forwardmamba}_ep300_lin90/log.txt`,
> "Max accuracy" 58.60 / 57.10). The report's Figure VI-2 quotes an earlier pair of Mamba probes
> (BiMamba 56.9, ForwardMamba 55.3); deploy the checkpoint whose number you intend to cite.

## What a linprobe checkpoint contains

A frozen 300-epoch encoder plus a **linear-probe head**:

```
head = nn.Sequential(
    nn.BatchNorm1d(768, affine=False, eps=1e-6),   # head.0 — the MAE "BN trick", no learnable scale/shift
    nn.Linear(768, 1000),                          # head.1 — the only trained parameters
)
```

The deployment tool detects a probe checkpoint by the presence of `head.1.*` keys and rebuilds the
`BatchNorm1d → Linear` head before loading, so the same tool serves both plain fine-tune heads and
linprobe heads (`scripts/demo_deploy_infer.py`, `load_checkpoint`). Load is `strict=False` and asserts
there are **no missing keys** — a shape/name drift fails loudly; the only unexpected keys are the
decoder/MAE-only tensors that a classifier does not use.

## Environment and files

The deployment tool and its per-arm CLS factories live in the research tree, **not** in this laptop
demo (the Mamba arms are CUDA/`mamba_ssm`-only and deliberately do not ship to a laptop):

- **Tool:** `scripts/demo_deploy_infer.py` (also in the main repo
  [`inference-efficient-convmae`](https://github.com/QuangCler/inference-efficient-convmae)).
- **Model factories:** `model_convmae_cls_{baseline,ghost,bimamba,forwardmamba}.py`
  (`convmae_baseline_cls` / `convmae_ghost_cls` / `convmae_bimamba_cls` / `convmae_forwardmamba_cls`,
  each built with `num_classes=1000`), plus the shared blocks (`blocks_ghost.py`,
  `blocks_mamba_{bidir,forward}.py`, `local_scan.py`, `conv_ffn.py`).
- **Where it runs (reference):** the A5000 box **gpu128**, directory `/root/capstone/linprobe_cls/`,
  virtualenv **`/root/mamba_venv2`** — **torch 2.8.0+cu128 with `mamba_ssm` and TensorRT 10.15 both
  importable**. This is the one environment where the Mamba PyTorch path *and* the base/ghost
  TensorRT path both work; the system Python there has `mamba_ssm` ABI-blocked, so always activate the
  venv first.

Preprocessing is the ImageNet eval transform — `Resize(256) → CenterCrop(224) → ToTensor →
Normalize(ImageNet mean/std)` — identical to how the probe was trained.

## Checkpoint sources

Point `--checkpoint` at the arm's 90-epoch probe:

| Arm | Where the deploy checkpoint comes from |
|---|---|
| baseline | W&B `convmae-linprobe` run `convmae_base_lineprobe_resume90_from_ep39` → `best_checkpoint.pth` |
| ghost | W&B `convmae-linprobe` run `allghost_ep300_linprobe_resume90_keep_lr` → `best_checkpoint.pth` |
| bimamba | gpu128 `/root/capstone/linprobe_cls/outputs/bimamba_ep300_lin90/best_checkpoint.pth` |
| forwardmamba | gpu128 `/root/capstone/linprobe_cls/outputs/forwardmamba_ep300_lin90/best_checkpoint.pth` |

The frozen backbones these probes sit on are the four 300-epoch pretrain checkpoints
(`convmae_base_pretrain_epoch300.pt`, `allghost_epoch_300.pt`,
`ghost_bimamba_pretrain_epoch300.pt`, `ghost_forwardmamba_pretrain_epoch300.pt`).

## 1. PyTorch deployment — all four arms

```bash
cd /root/capstone/linprobe_cls && source /root/mamba_venv2/bin/activate

# baseline / ghost — PyTorch (auto also builds a TensorRT engine, see §2)
python demo_deploy_infer.py --arm baseline \
    --checkpoint outputs/baseline_ep300_lin90/best_checkpoint.pth \
    --images $IMAGENET/val/n01440764 --limit 8 --pipeline pytorch --precision fp16

python demo_deploy_infer.py --arm ghost \
    --checkpoint outputs/ghost_ep300_lin90/best_checkpoint.pth \
    --images $IMAGENET/val/n01440764 --limit 8 --pipeline pytorch --precision fp16

# bimamba / forwardmamba — PyTorch ONLY (auto == pytorch for these arms)
python demo_deploy_infer.py --arm bimamba \
    --checkpoint outputs/bimamba_ep300_lin90/best_checkpoint.pth \
    --images $IMAGENET/val/n01440764 --limit 8 --pipeline pytorch --precision fp16

python demo_deploy_infer.py --arm forwardmamba \
    --checkpoint outputs/forwardmamba_ep300_lin90/best_checkpoint.pth \
    --images $IMAGENET/val/n01440764 --limit 8 --pipeline pytorch --precision fp16
```

Each run reports median batch latency and throughput and prints the top-5 class names per image
(`imagenet_class_index.json` if available, otherwise raw indices).

## 2. TensorRT deployment — baseline & ghost only

```bash
# base/ghost: exports ONNX (opset 17), builds an FP16 (or FP32) engine, runs it, then gates the
# result against PyTorch (max prob diff + top-1 agreement).
python demo_deploy_infer.py --arm baseline \
    --checkpoint outputs/baseline_ep300_lin90/best_checkpoint.pth \
    --images $IMAGENET/val/n01440764 --limit 8 --pipeline tensorrt --precision fp16

python demo_deploy_infer.py --arm ghost \
    --checkpoint outputs/ghost_ep300_lin90/best_checkpoint.pth \
    --images $IMAGENET/val/n01440764 --limit 8 --pipeline tensorrt --precision fp16
```

- `--pipeline auto` (the default) does the same for base/ghost: PyTorch first, then the TensorRT
  engine, then a PASS/CHECK gate comparing the two.
- **The Mamba arms will not build an engine.** `--pipeline tensorrt --arm bimamba` (or
  `forwardmamba`) prints that full TensorRT is unavailable for state-space arms (no standard ONNX
  selective-scan op) and stays on PyTorch. Do not try to force it.

> ⚠️ **TensorRT engines are not portable.** An engine is compiled for one specific GPU architecture +
> TensorRT/driver version and will fail to load on any other machine. Build on the target GPU; never
> copy `.engine`/serialized-network artifacts between different GPUs — rebuild instead. (Same caveat
> as the face-demo `build_trt.py` step.)

## Appendix — retraining a probe (reference recipe)

The probes were trained with `linprobe_custom.py` (on gpu128, in the `mamba_venv2` venv), not with
this repo's app. The recipe:

- **Frozen backbone**, LARS optimizer, head = `BatchNorm1d(affine=False) → Linear`.
- `--epochs 90`, `--blr 0.1` (`lr = blr × eff_batch / 256`), `--weight_decay 0.0`, `--warmup_epochs 10`.
- `--batch_size 128 --accum_iter 4` → **effective batch 512** (matches the upstream MCMAE probe).
- `--dist_eval`, `--data_path $IMAGENET/ILSVRC/Data/CLS-LOC`, `--wandb_project convmae-linprobe`.

```bash
python linprobe_custom.py --model <baseline|ghost|bimamba|forwardmamba> \
    --finetune models/<arm>_pretrain_epoch300.pt \
    --data_path $IMAGENET/ILSVRC/Data/CLS-LOC \
    --epochs 90 --blr 0.1 --weight_decay 0.0 --warmup_epochs 10 \
    --batch_size 128 --accum_iter 4 --dist_eval
```
