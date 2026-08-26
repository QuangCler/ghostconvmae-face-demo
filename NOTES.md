# Engineering notes

Working notes for this repository: the commands, the wiring that takes several files to see, and
the traps that have already cost time once. `README.md` is for someone running the demo; this is
for someone changing it.

## What this is

A self-contained Gradio demo that compares two backbones — **ConvMAE-Base** ("baseline") and
**Ghost+ConvMAE** ("ghost") — side by side on four face tasks (CelebA attributes, CASIA identity,
SCface low-res identity, LFW verification) **plus an ImageNet-1K linear-probe tab**. It is a
*demo/inference* package extracted from the research repo
[QuangCler/inference-efficient-convmae](https://github.com/QuangCler/inference-efficient-convmae):
no training code, no linter config, no packaging. `tools/selftest.py` is the only test.

Published at `QuangCler/ghostconvmae-face-demo` (code only — `datasets/`, `checkpoints/`, `engines/`
are gitignored and rebuilt by the scripts in `tools/`).

## Commands

```bash
# torch must be installed FIRST with the CUDA build matching the GPU (see README table);
# a CPU-only torch silently disables CUDA timing and TensorRT.
pip install -r requirements.txt

python tools/fetch_checkpoints.py     # ~6.2 GB of .pth into checkpoints/ (needs gdown)
python tools/fetch_datasets.py        # LFW+CelebA+CASIA into datasets/ (needs a kaggle token)
python tools/fetch_datasets.py --only imagenet   # ImageNet-1K VAL ONLY (~6.7 GB)
python tools/split_celeba.py          # CelebA flat folder -> datasets/celeba/{train,test}/
python tools/extract_casia_recordio.py           # RecordIO -> datasets/casia/prepared/{train,val}/

./run.sh   |   run.bat                # probes ./.venv then ./venv, always with -X utf8 -u
python app.py                         # same thing; serves http://127.0.0.1:7860 (0.0.0.0:7860)
DEMO_DEVICE=cpu python app.py         # PowerShell: $env:DEMO_DEVICE="cpu"; python app.py

python tools/build_trt.py                                 # 5 jobs x 2 arms x {fp32,fp16}
python tools/build_trt.py --jobs ImageNet --precisions fp16
```

`-X utf8` is not optional on Windows: picker labels contain `·` and the default cp1252 stdout
raises `UnicodeEncodeError` at import time.

**The test is `tools/selftest.py`.** It drives the *UI's own* `run_*` handlers, so what it prints is
what a button click produces:

```bash
python tools/selftest.py                                  # 5 jobs x 3 backends x 50 presses
python tools/selftest.py --jobs CASIA --iters 10 --skip-accuracy
python tools/selftest.py --samples 400 --out /tmp/r.json
```

Phase A presses "Run both models" N times per backend and reports which backend each arm *actually*
ran on, the resource row, top-1 stability across iterations, and the TensorRT-vs-PyTorch agreement
gate. Phase B scores a held-out sample with each task's own metric against the number the README
claims (`EXPECTED` in that file). A single forward is also enough for a quick check:
`python -c "import app,torch; print(app.infer_one('ImageNet','ghost',torch.zeros(1,3,224,224).to(app.DEVICE),'PyTorch')[1:])"`

## Architecture

```
app.py          UI + inference        vendor/convvit/   face architecture (flat imports)
demo/           library code          vendor/linprobe/  linprobe architecture (flat imports)
tools/          one-off scripts       assets/           class maps, reference numbers
```

**Everything is driven by the `JOBS` dict in [app.py](app.py)** — one entry per task holding its
`task` kind (`attribute` / `identity` / `verification` / `classification`), class count, checkpoint
filenames, dataset folder, and UI copy. `tools/build_trt.py` keeps a *parallel, smaller* `JOBS`
dict; adding or renaming a task means editing both, and the engine filename convention
`{job}_{arm}_{prec}.engine` is the contract between them ([demo/trt_backend.py](demo/trt_backend.py)).

Three layers, thin and in this order:

1. **Model builders.** [demo/face_models.py](demo/face_models.py) — `convvit_base` / `convvit_ghost`
   construct the same `ConvViT` (img_size `[224,56,28]`, embed_dim `[256,384,768]`, depth
   `[2,2,11]`); the ghost arm only swaps stage-1/2 `CBlock`s for `GhostV2BlockMasked`.
   [demo/linprobe_models.py](demo/linprobe_models.py) — a **different** network for the same weights:
   the linprobe classifier fuses `stage1_output_decode`/`stage2_output_decode` into stage 3 and
   applies `norm` *before* the mean-pool, where the face `ConvViT` mean-pools then applies `fc_norm`.
   Don't unify them. Both loaders strip `module.` prefixes, load `strict=False`, and **assert there
   are no missing keys** — a shape/name drift fails loudly here.
2. **Backend dispatch** ([`infer_one`](app.py)) — the single choke point for inference. Selects
   PyTorch or a TensorRT engine, warms up `_WARMUP` times per `(job, arm, backend, feats)`, then times the one
   real forward and captures its activation peak, returning `(out, median_ms, backend_label, meta)`.
   TensorRT is *always* best-effort: missing engine, missing `tensorrt`, or any exception at run time
   falls back to PyTorch and the label in the results table records what actually ran. Warm-up sits
   *inside* that guard, and the fallback is warmed again before it is timed — see the bite below.
3. **Gradio UI** ([`build_ui`](app.py)) — CelebA tab, a reusable `identity_tab()` factory for
   CASIA/SCface, a two-stage person→photo LFW tab, and the ImageNet linear-probe tab. Every
   component is appended to `clearables`, and each tab's `.select` resets all of them, so switching
   tabs wipes prior inputs and results.

`feats=True` routes through `embed()` → `forward_features` for the 768-d LFW embedding;
`tools/build_trt.py` mirrors this at export time with the `EmbedOnly` wrapper, which is why LFW
engines emit an embedding rather than logits.

**Dataset layout is declared, not sniffed** ([demo/dataset_paths.py](demo/dataset_paths.py)). An
earlier version probed directory trees and scored them, which silently mapped SCface onto its `cam_*`
folders and CASIA onto whatever happened to be extracted. Everything is explicit now, behind four
functions: `test_samples(job)` (the held-out picker index), `class_label(job, i)` / `class_face(job, i)`
(naming a predicted class and finding a train image of it), and `locate(job, filename)` (which split a
dropped image came from). Adding a dataset means adding entries there, not teaching a heuristic.

## Things that bite

- **`vendor/` uses flat top-level imports.** `models_convvit.py` does `import vision_transformer`, so
  `demo/face_models.py` puts `vendor/convvit/` on `sys.path` and imports flat; `demo/linprobe_models.py`
  does the same with `vendor/linprobe/`. Never rewrite these as package imports — the checkpoints and
  the bundled files depend on this layout. Both directories carry byte-identical copies of
  `vision_transformer.py` / `blocks_ghost.py`; that duplication is deliberate (each mirrors the
  research repo as shipped).
- **Scripts under `tools/` compute `HERE` as the *parent* of their own directory** and insert the repo
  root on `sys.path` before `from demo import …`. A new tool that copies the old flat-layout
  `HERE = dirname(abspath(__file__))` will silently write into `tools/datasets/`.
- **CASIA and SCface have `num_classes: None`** and infer the head size from `head.weight` in the
  checkpoint (`_head_size`). Don't hardcode a class count for them. ImageNet is fixed at 1000 and its
  head is `Sequential(BatchNorm1d(affine=False), Linear)` — detected by `head.1.*` keys and rebuilt
  before load, so `model.head.weight` does not exist for that job.
- **Preprocessing must stay exactly** Resize 256 **bicubic** → CenterCrop 224 → ImageNet normalise
  (`PREPROC` in app.py). Bilinear was a real accuracy regression, worst on SCface. The same transform
  is correct for the linear probe (it is the ImageNet eval transform the probe trained under).
- **CASIA identity comes from the RecordIO IRHeader label, never from `train.lst`.** The two are
  different lengths and orders (501,196 records vs 494,149 lines) and drift apart through the file;
  pairing them by position mislabels every image and drops held-out top-1 from ~91% to 1%. Class
  indices are named via `datasets/casia/class_order.json`, falling back to
  `assets/casia_classes.json` — which is what a fresh clone gets, since `datasets/` is gitignored.
  Both are `ImageFolder(train_dir).classes` from the fine-tune run itself; do not reconstruct that
  ordering from folder names. One list covers both arms: the ordering depends only on the raw
  RecordIO dump, not on which run consumed it, and the two arms' exports were byte-identical.
- **SCface's `surveillance_cameras_all/` is three populations, and the report's number is one of
  them.** `cam1`-`cam5` × 3 distances are visible surveillance (1,950 imgs) — this is what the
  paper's 45.13% / 31.36% refers to; `cam6`/`cam7` are the **infrared** cameras (780 imgs); `cam8`
  is an IR **mugshot** despite the folder name (130 imgs). Scoring all three pooled reads far below
  the paper and looks like a regression that is not one — the model never trained on infrared.
  Report the visible cameras only, and quote the paper's numbers, not a locally measured pool: the
  UI copy, the README and `selftest.py` all state the report's figures. The picker labels IR, and
  `selftest.py` scores SCface on `top-1 visible`. Also: `cam8` files carry no distance token
  (`001_cam8.jpg`), so any `len(parts) >= 3` parse silently drops those 130.
- **A TensorRT row's Weights/Peak VRAM come from the engine, not from torch.** TensorRT holds its
  weights and scratch outside the torch allocator, so measuring an engine the PyTorch way reports
  the idle *PyTorch* copy's weights and a near-zero activation peak. `trt_backend.footprint()`
  reads `device_memory_size` and the engine file size instead. Likewise the LFW row sums its two
  batch-1 engine calls rather than averaging them, so it stays comparable with the PyTorch row's
  single batch-2 forward — averaging made TensorRT look ~2x better than it is.
- **ImageNet ground truth is the wnid folder name**, because wnids in sorted order *are* the class
  indices `ImageFolder` gave the probe. So `fetch_datasets.py --only imagenet` pulls the val-only
  mirror `titericz/imagenet1k-val` (~6.7 GB) rather than the Kaggle competition, whose ~155 GB archive
  bundles a train half this demo never touches and extracts val flat, i.e. without labels.
- **LFW's verdict cut is `LFW_THRESHOLD = 0.235`, not 0.5.** At 0.5 the embeddings separated fine
  (ROC-AUC 0.9925 / 0.9809) but the cut sat far too high: TPR 49.5% / 37.5% at TNR 100%, so genuine
  pairs read "DIFFERENT". 0.235 gives ~95% / ~92% accuracy on the same sample.
- **TensorRT engines are not portable** across GPU architecture / TensorRT version. `engines/` is
  gitignored; on a new machine rebuild with `tools/build_trt.py`, never copy `.engine` files. This
  bites *on the same machine* too: `engines/` was built by `./.venv` (TensorRT 10.16), and the
  system Python here carries TensorRT 11.2, so `python app.py` outside the venv fails **every**
  engine at once. `deserialize_cuda_engine` does not raise on a version mismatch — it logs and
  returns `None` — so `_load` checks for `None` and raises with the version in the message, and
  `trt_backend._BROKEN` remembers the verdict rather than re-deserializing on every click.
- **A PyTorch fallback has to be warmed before it is timed, and warm-up has to be inside the
  try.** Both halves of `infer_one`'s guard were wrong and both showed up as a Base-vs-Ghost
  latency gap of *many times*, which no pair of these models can produce (the real spread is
  1.05x-1.26x). A broken engine fails on its first forward, which is a warm-up one — outside the
  guard, that raised straight out of the click handler. And when the failure landed on the timed
  call instead, the old code timed `torch_call` cold: **251 ms against a 41 ms warm forward here,
  6.1x**, sat next to the other arm's 23 ms engine number. `selftest.py` now gates on
  `PAIR_RATIO_MAX = 1.6` and on both arms reporting the same `ran=` label.
- **Two app instances on one 4 GB card is the other way to get nonsense latencies.** `app.py`
  does not check the port itself — Gradio does, and a second `python app.py` can sit there holding
  ~3.5 GB of models without ever serving. If timings look wild, check `nvidia-smi` and
  `Get-CimInstance Win32_Process -Filter "Name like 'python%'"` before touching the code.
- **Untrained-checkpoint guard:** `get_model` flags a head with σ < 0.005 as `untrained` and the UI
  shows a warning (`_warn`). This exists because a clobbered SCface file once produced uniform
  "random" predictions; `fetch_checkpoints.py` now pins md5s so a bad local copy is re-fetched.
- **Peak VRAM is per-model, not process-wide** (`_table`). It is the model's own weights plus the
  activation rise of the timed forward. A bare `max_memory_allocated()` counts the other cached arm
  and the CUDA context, which reports the same ~700 MB for both arms and erases the comparison the
  demo exists to make. Latency is the **median of the last `_LAT_WINDOW` real forwards** — a single
  batch-1 timing swings 38-77 ms here, far more than the ~10% gap between the arms. Neither
  measurement runs an extra pass.
- **Ten backbones technically fit in 4 GB, and that is the problem.** All ten load (3,447 MB
  allocated, 3,829 MB reserved) but leave **0 MB free**, so a visitor who has opened every tab and
  then picks TensorRT has no room for an engine and is silently dropped back to PyTorch. So
  `_CACHE` holds **one job only** — both arms of the tab you are on, nothing else. `_release(job)`
  drops every other job's models *and*, through `trt_backend.release(job)`, its engines. It is
  called from each tab's `.select`, so the card is freed the moment you leave a tab, *and* at the
  top of `get_model`, which is what makes it hold for anything driving the handlers directly
  (`selftest.py`) rather than through a browser event. Measured touring all five tabs: never more than 2 models
  and 2 engines, 1.0-2.6 GB free throughout, against 0 MB before. Two things this must keep doing:
  clear the evicted models' `_WARMED` flags, or the reloaded copy goes straight to a timed forward
  while still cold; and free the *engines*, which are the larger half — TensorRT keeps weights and
  scratch outside torch's allocator where `empty_cache()` cannot reach them. Engines are filtered
  by job, not by precision, so toggling FP16/FP32 inside one tab keeps both and stays instant.
- **Gradio 6 moved `theme`/`css`/`js` from `gr.Blocks(...)` to `.launch(...)`.** `_style_kwargs` picks
  the right one by inspecting the signature; passing them to the wrong place is only a `UserWarning`,
  after which the app renders unstyled and the dark-mode script never runs.
- **`.env` holds live W&B / GitHub PAT / Kaggle credentials.** It is in `.gitignore`, but the values
  are real — never echo them into output or a commit, and treat them as compromised if this folder is
  shared. The PAT's scopes are broad (`repo`, `admin:org`, `delete_repo`, `workflow`).
