# GhostConvMAE — Fine-tuned Face Model Demo

Interactive demo comparing the two deployed backbones — **ConvMAE-Base** vs **Ghost+ConvMAE** —
side by side on **four face tasks plus an ImageNet-1K linear-probe tab**. Each tab has a **short
description of what the models do**, an **in-UI image picker** (choose a sample straight from that
task's dataset, or drop your own — the chosen filename is shown), and the two models' predictions
next to each other. You can switch the inference backend between **native PyTorch**
and a **TensorRT engine** in the UI. Runs on a laptop: a small CUDA GPU (e.g. **GTX 1650, 4 GB**) or
plain **CPU** (auto-detected). Switching tabs clears the previous tab's inputs and results.

**Only the two convolution/Transformer arms (ConvMAE-Base, Ghost+ConvMAE) are shown — in every
tab, including the linear-probe one.** The Ghost+ForwardMamba and Ghost+BiMamba arms are
deliberately excluded from this laptop demo: their Stage-3 Mamba blocks need the CUDA-only
`mamba_ssm` selective-scan kernels, which do not install on the target GTX 1650, and the
selective-scan op has no ONNX/TensorRT path — so those arms can neither run in PyTorch here nor be
exported to an engine.

> **The full four-arm linear-probe deployment** (all four arms, on a CUDA server) is documented
> separately in **[LINPROBE_DEPLOY.md](LINPROBE_DEPLOY.md)** — there baseline/ghost run through
> **PyTorch + TensorRT** and the two Mamba arms through **PyTorch only**.

## Tasks
- **CelebA — attributes**: pick/drop a face → **every attribute the model scores above 50%**, most
  confident first. Attribute recognition is multi-label and the number of true attributes varies per
  face, so a fixed top-5 both hid real positives and padded plain faces with sub-50% guesses; if
  nothing clears the bar the top-3 are shown instead. When the image belongs to CelebA, its
  **ground-truth attributes** are listed and each model is **scored against them**: the share of the
  40 binary calls it gets right, plus F1 on the positives — accuracy alone is flattered by the many
  attributes that are negative for nearly every face.
- **SCface — cross-resolution identity (top-5)**: pick/drop a face → top-5 subject bars **plus the
  top-1 subject's mugshot**, i.e. the image the model actually trained on, so you see *who* the
  number is. The 50% threshold decides the verdict wording ("predicted" vs "no confident match"),
  not whether the face appears. It is the low-res stressor (trained on mugshots, tested on
  surveillance crops), and `surveillance_cameras_all/` is **three populations**, which the picker
  now labels: 1,950 **visible** crops (`cam1`–`cam5`) — the report's setting, and the entries left
  unmarked — 780 marked **`IR`** (`cam6`–`cam7`), and 130 marked **`IR mugshot`** (`cam8`). The
  model never saw infrared, so a marked entry is a curiosity, not the benchmark. Most crops read as
  low confidence either way.
- **CASIA — identity (top-5)**: pick/drop a face → top-5 identity bars plus the predicted person's
  face, taken from the train split. Paper (3 seeds): Top-1 **91.49%** (base) / **91.32%** (ghost).
  Both identity tabs carry a **"Where the faces come from"** panel.
- **LFW — verification**: for each side pick a **person** (type to search), then one of **their
  photos**; the two faces give a cosine similarity + same/different verdict per model. A checkbox
  limits the list to the **1,680 people with ≥2 photos** (of 5,749) — someone with a single photo
  can only ever form a different-person pair, so they cannot demonstrate a match. That 1,680 is the
  same count as the fine-tune's LFW head. Verification is open-set, so no LFW identity is a
  training class and any two photos are a valid test.
- **ImageNet-1K — linear probe (top-5)**: pick/drop any image → each backbone's top-5 of the 1,000
  ImageNet classes. This is the **representation-quality probe**: the backbone is **frozen** and only
  a linear head (BatchNorm→Linear) was trained for **90 epochs** on top of each **300-epoch pretrained**
  backbone — no fine-tuning. Reported linear-probe Top-1 is **64.06%** (Base) / **58.60%** (Ghost).
  Base/Ghost only, for the reason above. Val images carry their wnid folder, so the panel also
  scores each backbone against the true class (top-1 / top-5 / missed).

### What each picker offers

[dataset_paths.py](dataset_paths.py) declares the layout per dataset. All held-out data, so what you
see is a fair test:

| Task | Offered in the picker | Trained on (not offered) |
|---|---|---|
| CelebA | `celeba/test/` — partitions **1 (val) + 2 (test)**, 39,829 | `celeba/train/` — partition 0, 162,770 |
| CASIA | `prepared/val/<label>/` — 2 images per identity | `prepared/train/<label>/` — the predicted face |
| SCface | `surveillance_cameras_all/` — visible crops + `IR` + `IR mugshot`, labelled | `mugshot_frontal_cropped_all/` — the predicted face |
| LFW | any photo of any of the 5,749 people | nothing; verification is open-set |
| ImageNet | `imagenet/imagenet-val/<wnid>/` — the 50,000 held-out val images, folder = ground truth | `train/` — not shipped; the probe's backbone is frozen anyway |

Each dropdown loads 300 entries and pages in 300 more when you scroll its list to the bottom (or
click the **Load 300 more** button under it). Embedding a whole split instead — CelebA's is ~40k —
inflates the served page from 3 MB to 24 MB, since every choice ships in the payload.

CelebA ships as one flat folder of 202,599 images; split it so the two halves are separate on disk:

```bash
python tools/split_celeba.py          # -> datasets/celeba/{train,test}/, per list_eval_partition.csv
python tools/split_celeba.py --check  # report only, change nothing
```

Files are moved rather than copied, and anything already in place is left alone, so it resumes
after an interruption. Without the split the tab still works — it filters the flat folder through
the partition file instead.

### Unpacking CASIA correctly

CASIA ships as an InsightFace RecordIO, not as JPEGs, so it must be unpacked before the identity tab
works:

```bash
python tools/extract_casia_recordio.py     # -> datasets/casia/prepared/{train,val}/<label:05d>/
```

**A record's identity comes from its IRHeader label — never from `train.lst`.** An earlier version
of this script paired record N with line N of `train.lst`, and that quietly corrupted every label:
`train.idx` holds 501,196 records against `train.lst`'s 494,149 lines, in a different order. They
agree at the start and drift apart — at key 5,000 the record header says label 28 while line 5,000
says 27; by key 341,263 it is 6,332 against 6,285. Every image landed in a neighbouring identity's
folder, so the predicted-person face was wrong and held-out top-1 measured **1%**. Reading the label
from the record header instead gives **90.8%** (base) / **90.5%** (ghost), in line with the paper.

Two more traps this script now handles: the Kaggle zip nests its payload one level down, and a
half-finished unzip leaves a **truncated `train.rec` beside the complete one** (1.88 GB vs 2.73 GB)
— reading the short file silently drops 40% of the identities, so `--raw` picks whichever copy is
long enough to cover its own index. Rerunning skips images already written, so it resumes safely.

Class indices are named through `datasets/casia/class_order.json`, which is
`ImageFolder(train_dir).classes` from the fine-tune run itself (kept in
[assets/casia_class_maps/](assets/casia_class_maps/)) — no reconstruction. Folder names are the
RecordIO integer labels, and they are **not contiguous**: label `09282` has only 2 images, so both
went to val and it never reached `train/`, which is why the head is 10,571 and class index `i` maps
to label `i` below 9282 and `i+1` from 9282 on. `sorted(os.listdir(prepared/train))` reproduces that
list exactly — worth re-checking if you ever re-unpack.

> **Rendering:** the app forces Gradio into dark mode (it appends `?__theme=dark`). On Gradio 6 the
> `theme`/`css`/`js` arguments moved from `gr.Blocks(...)` to `.launch(...)`; `app.py` detects which
> signature is in use, so the styling applies on Gradio 4/5 and 6 alike.

Each result also shows a **resource table**: backend, params, weights, **latency**, **measured peak
VRAM**, and the paper's A5000 reference, plus a Ghost-vs-Base delta row — see
[Latency & peak VRAM](#latency--peak-vram) for what the two VRAM columns mean.

## 1. Install PyTorch for your GPU (do this first)

Install the torch build that matches your NVIDIA GPU / driver, then the rest. If you skip this and
`pip install torch` gives a CPU-only build, the app runs on CPU and cannot measure VRAM or use TensorRT.

| GPU (arch) | Recommended install | Notes |
|---|---|---|
| **GTX 16xx / RTX 20xx** (Turing, e.g. **RTX 1650**) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` | No Tensor Cores → FP16 not accelerated, but valid; ~0.3–1.4 GB/model |
| **RTX 30xx** (Ampere) | `... /whl/cu121` | Tensor Cores → FP16/TF32 fast |
| **RTX 40xx** (Ada) | `... /whl/cu124` (or cu121) | |
| **A5000 / A100** (data-center) | `... /whl/cu124` | Matches the paper's benchmark GPU |
| **No NVIDIA GPU** | `pip install torch torchvision` | CPU only — VRAM/TensorRT disabled |

Verify: `python -c "import torch; print(torch.cuda.is_available())"` must print **True**.

```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
# 1) install the torch build from the table above, then:
pip install -r requirements.txt
pip install gdown
python tools/fetch_checkpoints.py         # face checkpoints (~5.5 GB) -> ./checkpoints/
```

## 2. (Recommended) download the original datasets — they power the in-UI picker

```bash
pip install kaggle                  # then put your token at ~/.kaggle/kaggle.json
python tools/fetch_datasets.py            # LFW + CelebA + CASIA -> ./datasets/  (large!)
python tools/fetch_datasets.py --only lfw # just one
python tools/fetch_datasets.py --only imagenet   # ImageNet-1K VAL ONLY (~6.7 GB) for the linear-probe tab
```
Each tab's **image picker reads from `datasets/<task>/`**, so downloading a set fills that tab's
dropdown with real samples to choose from. Without them you can still drop your own image into any
tab. SCface is access-controlled and is not downloaded (request a licence, then place the images
under `datasets/scface/` — e.g. the prepared `val/<id>/*` surveillance crops — and they appear in
the SCface picker).

**ImageNet (linear-probe tab).** The picker reads the standard **ImageNet-1K val** set (50,000
held-out images). `--only imagenet` pulls **only the validation split** — the mirror
`titericz/imagenet1k-val`, ≈ 6.7 GB — deliberately *not* the Kaggle competition
`imagenet-object-localization-challenge`, whose single ~155 GB archive bundles a train half this
demo never touches (the probe's backbone is frozen and is only ever evaluated on val).

Images arrive as `<wnid>/ILSVRC2012_val_*.JPEG`, and **that folder name is the ground truth**:
wnids in sorted order are exactly the class indices `ImageFolder` gave the probe at training time,
so the tab marks each backbone ✓ top-1 / ✓ in top-5 / ✗ missed with no label file involved. A flat
folder of `*.JPEG` also works, just without the scoring. To reuse a copy you already have, set
`IMAGENET_VAL_DIR=/path/to/val` and skip the download entirely. Class names come from the bundled
`imagenet_class_index.json`.

## 3. (Optional) build TensorRT engines — **on your own GTX 1650**

> ⚠️ **Engines are not portable.** A TensorRT engine is compiled for one specific GPU
> architecture + TensorRT/driver version. An engine built on any other machine (an A5000, a
> friend's RTX 30xx, this repo's server, …) **will fail to load on your GTX 1650**. So the demo
> ships **no** pre-built engines — you must run `build_trt.py` **on the GTX 1650 itself**. Never
> copy `.engine` files between different GPUs; rebuild instead.
>
> The version half of that bites on one machine too: **run the app with the same interpreter that
> built the engines**. If `engines/` came from a venv on TensorRT 10 and you then start the app
> with a system Python carrying TensorRT 11, every engine fails to deserialize at once and every
> row falls back to PyTorch. Use `run.sh` / `run.bat`, which pick `./.venv` for you.

**3a. Install TensorRT** (matching the CUDA your torch uses — e.g. cu121 from step 1):

```bash
pip install onnx
pip install tensorrt                 # pulls the TensorRT 10.x wheels (cu12)
# Windows: same pip command works in the venv. If `import tensorrt` fails, install the
# TensorRT zip from developer.nvidia.com/tensorrt matching your CUDA, and add its lib/ to PATH.
python -c "import tensorrt, torch; print(tensorrt.__version__, torch.cuda.get_device_name(0))"
```

**3b. Build the engines on the GTX 1650** (takes ~1–3 min each; writes to `./engines/`):

```bash
python tools/build_trt.py                                  # everything: 5 tasks x 2 arms x {fp32,fp16}
python tools/build_trt.py --jobs ImageNet --precisions fp16    # just the linear-probe engines
python tools/build_trt.py --jobs CelebA LFW --precisions fp16  # or just what you need (faster)
```

`ImageNet` is one of the jobs (Base/Ghost linear-probe classifiers) and builds like the others; the
Mamba arms are not a job here — they have no ONNX/TensorRT path.

It prints the GPU it is building on — make sure that is your **GTX 1650**. Classification tasks
export encoder+head (→ logits); LFW exports the encoder (→ 768-d embedding). Workspace is capped
at 1 GB so it fits the 4 GB card.

**3c.** Launch the app (step 4) and pick **TensorRT (FP16)** / **(FP32)** in the backend dropdown.
A model with no engine, an unimportable `tensorrt`, or an engine that will not deserialize falls
back to PyTorch — and the **Backend** column of the resource table says so, e.g.
`PyTorch (no engine)` or `PyTorch (TRT failed: RuntimeError)`. Read that column before reading the
latency: a fallback is a PyTorch number sitting in a TensorRT run.

## 4. Run

```bash
python app.py                       # then open http://127.0.0.1:7860
```
- **Backend dropdown** (top of the page): `PyTorch`, `TensorRT (FP16)`, or `TensorRT (FP32)`.
  TensorRT options use the engines from step 3; if an engine is missing or TensorRT isn't
  installed, that model transparently falls back to PyTorch.
- Force CPU: `DEMO_DEVICE=cpu python app.py` (PowerShell: `$env:DEMO_DEVICE="cpu"; python app.py`).

## 5. (Optional) check the whole thing

```bash
python tools/selftest.py                                  # 5 tasks x 3 backends x 50 presses
python tools/selftest.py --jobs CASIA --iters 10 --skip-accuracy
```

It drives the tabs' **own** "Run both models" handlers, so what it prints is what a click produces.
For each task and backend it reports which backend each arm *actually* ran on (TensorRT falls back
silently, and the label is the only honest record), the resource row, whether top-1 held steady
across all iterations, and the TensorRT-vs-PyTorch agreement gate. Then it scores a held-out sample
with each task's own metric against the number the report claims. The pass band is the stated
tolerance **plus two standard errors of the subset you actually ran** — `--samples 200` on a ~64%
top-1 carries a 3.4-point standard error of its own, so a fixed gate would fail perfectly ordinary
draws. Measured on a GTX 1650:

| Task | Metric | Base | Ghost | Claimed |
|---|---|---|---|---|
| CelebA | mAP (400 imgs) | 0.783 | 0.780 | 0.789 / 0.778 |
| CASIA | top-1 (400) | 91.3% | 88.8% | 91.49% / 91.32% |
| SCface | top-1, visible cams (279) | 45.9% | 31.5% | 45.13% / 31.36% |
| LFW | ROC-AUC (400 pairs) | 0.995 | 0.986 | 0.9921 / 0.9833 |
| ImageNet | top-1 (400) | 66.5% | 60.3% | 64.06% / 58.60% |

All 30 task/arm/backend combinations run, every engine loads, and TensorRT agrees with PyTorch on
top-1 in every case (max probability difference 0.004 at FP16, 0.0000 at FP32).

It also gates the **Base-vs-Ghost latency ratio**, because the two arms are one network with two
stages swapped and cannot be far apart: measured across all five tasks and all three backends the
spread is **1.05x–1.26x**, and the check fails past 1.6x or if the two arms report different
backends. A many-fold gap has never been the models — it is one arm quietly falling back to
PyTorch while the other runs on an engine.

## Latency & peak VRAM

Both are measured on the **single forward that produced the prediction you are looking at** — the
demo adds no extra passes, so turning the table on costs nothing at inference time.

- **Latency** — CUDA events around that forward (wall-clock `perf_counter` on CPU). CUDA events
  measure the GPU's own execution window rather than the wall time around an asynchronous launch.
  Each model runs a few warm-up forwards per task and backend before its first *timed* one, so
  the figure is always a warm number; the first click of a tab simply takes a moment longer to come
  back. Two warm-ups turned out not to be enough — measured over 50 presses with a different image
  each time, the first timed forward still landed +16% over steady state on ImageNet/PyTorch, which
  is exactly the number someone reads after a single click. It is 5 now (+2%); TensorRT never showed
  the effect, since an engine does no run-time autotuning.
- **Peak VRAM** — that model's **weights + the activation peak of that same forward**, taken as the
  rise in `max_memory_allocated` over the pre-forward baseline. Reporting the raw process-wide peak
  instead — what a naive `max_memory_allocated()` call gives — counts the *other* cached arm and the
  CUDA context, inflating both models to roughly the same ~2 GB and hiding the very difference this
  demo exists to show. On CPU the column reads `— (CPU)`; if your NVIDIA GPU shows that, your
  PyTorch is a CPU-only build. On a **TensorRT** row the two columns come from the engine instead —
  its file size and the scratch block it requests — because TensorRT keeps both outside torch's
  allocator, so measuring an engine the PyTorch way would report the idle PyTorch copy sitting in
  the cache rather than what actually ran.
- **Paper A5000** — the report's **FP16, batch-32** figure from `resource_meta.json` (Base 771 /
  Ghost 609 MB). Batch 32 against batch 1 is why it is larger: use it as the reference regime, and
  use the measured column to compare the two arms on your own machine.

On the **LFW** tab the latency covers the **pair**: PyTorch embeds both faces in one batch-2
forward, and because engines are built for batch 1, the TensorRT row *sums* its two calls. Averaging
them instead would put a per-face number beside a per-pair one and show TensorRT as ~2x faster than
it is (measured on a GTX 1650: 66 ms PyTorch vs 47 ms TensorRT-FP16 for the pair).

## Notes
- `casia_baseline.pth` is a freshly retrained demo checkpoint; the report's CASIA numbers are unchanged.
- SCface checkpoints: **base = seed1**, **ghost = seed0** — the two seeds behind the report's
  45.13% / 31.36% on the visible cameras. *(seed0's base
  file on Drive had been clobbered to an untrained epoch-0 state, which made ConvMAE-Base output
  uniform "random" predictions; `fetch_checkpoints.py` now pins the correct seed1 file and its md5, so
  a previously-downloaded broken copy is re-fetched.)*
  Base's top-1 is often wrong, so read the top-5, and prefer a *visible* surveillance crop: an entry
  marked `IR` is a different sensor the model never trained on. Predictions are subject IDs `001–130` matching the
  filename's leading number. The app also self-checks each classification checkpoint and shows a
  warning if a loaded head looks untrained, so a bad file never silently degrades to random again.
- Preprocessing matches the fine-tune **eval transform exactly** (Resize 256 **bicubic** → CenterCrop
  224 → ImageNet normalise); using bilinear here previously degraded predictions, worst on SCface.

## Layout
```
demo_app/
  app.py                        Gradio UI + inference (PyTorch / TensorRT backend selector)
  demo/                         the app's own library code
    dataset_paths.py            where every split lives; the only place layout is declared
    face_models.py              base/ghost FACE model builders + checkpoint loader
    linprobe_models.py          base/ghost IMAGENET linear-probe builders + loader
    trt_backend.py              TensorRT engine loader + runtime
  vendor/                       research model code, imported flat (do not repackage)
    convvit/                    face architecture (ConvViT, Ghost blocks)
    linprobe/                   linprobe architecture (MAE + CLS factories, Ghost blocks)
  tools/                        one-off scripts, run as `python tools/<name>.py`
    fetch_checkpoints.py        downloads checkpoints into checkpoints/
    fetch_datasets.py           downloads the original datasets into datasets/ (Kaggle)
    extract_casia_recordio.py   CASIA RecordIO -> prepared/{train,val}/<label>/
    split_celeba.py             CelebA flat folder -> {train,test}/
    calibrate_lfw.py            refit the LFW cosine threshold on the official pairs
    build_trt.py                export ONNX + build TensorRT engines into engines/
    selftest.py                 every task x arm x backend, through the demo's own buttons
  assets/
    imagenet_class_index.json   ImageNet-1K class index -> readable top-5 names
    resource_meta.json          reference A5000 resource numbers
    casia_class_maps/           the fine-tune's own CASIA class ordering (+ its prepare script)
  docs/LINPROBE_DEPLOY.md       four-arm linear-probe deployment on a CUDA server
  README.md  NOTES.md  requirements.txt  run.sh  run.bat
  checkpoints/  engines/  datasets/   (populated by the scripts above)
```
