# GhostConvMAE — Fine-tuned Face Model Demo

Interactive demo comparing the two deployed backbones — **ConvMAE-Base** vs **Ghost+ConvMAE** —
on four face tasks, side by side. Each tab has a **short description of what the models do**, an
**in-UI image picker** (choose a sample straight from that task's dataset, or drop your own — the
chosen filename is shown), and the two models' predictions next to each other. You can switch the
inference backend between **native PyTorch**
and a **TensorRT engine** in the UI. Runs on a laptop: a small CUDA GPU (e.g. **RTX 1650, 4 GB**) or
plain **CPU** (auto-detected). Switching tabs clears the previous tab's inputs and results.

Only the two fine-tuned convolution/Transformer arms are included; the Mamba arms are not part of
the fine-tuning demo (their CUDA-only kernels don't ship to a laptop).

> **Deploying the ImageNet-1K linear-probe classifiers** (the 90-epoch linear heads on the frozen
> 300-epoch pretrained backbones, all four arms) is documented separately in
> **[LINPROBE_DEPLOY.md](LINPROBE_DEPLOY.md)** — baseline/ghost run through **PyTorch + TensorRT**,
> the two Mamba arms through **PyTorch only**.

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
  surveillance crops — Base Top-1 ~44%, so most crops read as low confidence).
- **CASIA — identity (top-5)**: pick/drop a face → top-5 identity bars plus the predicted person's
  face, taken from the train split. Held-out top-1 measured here is **90.8%** (base) / **90.5%**
  (ghost), matching the paper. Both identity tabs carry a **"Where the faces come from"** panel.
- **LFW — verification**: for each side pick a **person** (type to search), then one of **their
  photos**; the two faces give a cosine similarity + same/different verdict per model. A checkbox
  limits the list to the **1,680 people with ≥2 photos** (of 5,749) — someone with a single photo
  can only ever form a different-person pair, so they cannot demonstrate a match. That 1,680 is the
  same count as the fine-tune's LFW head. Verification is open-set, so no LFW identity is a
  training class and any two photos are a valid test.

### What each picker offers

[dataset_paths.py](dataset_paths.py) declares the layout per dataset. Three of the four tabs offer
held-out data only, so what you see is a fair test:

| Task | Offered in the picker | Trained on (not offered) |
|---|---|---|
| CelebA | `celeba/test/` — partitions **1 (val) + 2 (test)**, 39,829 | `celeba/train/` — partition 0, 162,770 |
| CASIA | `prepared/val/<label>/` — 2 images per identity | `prepared/train/<label>/` — the predicted face |
| SCface | `surveillance_cameras_all/` low-res crops | `mugshot_frontal_cropped_all/` — the predicted face |
| LFW | any photo of any of the 5,749 people | nothing; verification is open-set |

Each dropdown loads 300 entries and pages in 300 more when you scroll its list to the bottom (or
click the **Load 300 more** button under it). Embedding a whole split instead — CelebA's is ~40k —
inflates the served page from 3 MB to 24 MB, since every choice ships in the payload.

CelebA ships as one flat folder of 202,599 images; split it so the two halves are separate on disk:

```bash
python split_celeba.py          # -> datasets/celeba/{train,test}/, per list_eval_partition.csv
python split_celeba.py --check  # report only, change nothing
```

Files are moved rather than copied, and anything already in place is left alone, so it resumes
after an interruption. Without the split the tab still works — it filters the flat folder through
the partition file instead.

### Unpacking CASIA correctly

CASIA ships as an InsightFace RecordIO, not as JPEGs, so it must be unpacked before the identity tab
works:

```bash
python extract_casia_recordio.py     # -> datasets/casia/prepared/{train,val}/<label:05d>/
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
[advice/casia_class_maps/](advice/casia_class_maps/)) — no reconstruction. Folder names are the
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
python fetch_checkpoints.py         # face checkpoints (~5.5 GB) -> ./checkpoints/
```

## 2. (Recommended) download the original datasets — they power the in-UI picker

```bash
pip install kaggle                  # then put your token at ~/.kaggle/kaggle.json
python fetch_datasets.py            # LFW + CelebA + CASIA -> ./datasets/  (large!)
python fetch_datasets.py --only lfw # just one
```
Each tab's **image picker reads from `datasets/<task>/`**, so downloading a set fills that tab's
dropdown with real samples to choose from. Without them you can still drop your own image into any
tab. SCface is access-controlled and is not downloaded (request a licence, then place the images
under `datasets/scface/` — e.g. the prepared `val/<id>/*` surveillance crops — and they appear in
the SCface picker).

## 3. (Optional) build TensorRT engines — **on your own GTX 1650**

> ⚠️ **Engines are not portable.** A TensorRT engine is compiled for one specific GPU
> architecture + TensorRT/driver version. An engine built on any other machine (an A5000, a
> friend's RTX 30xx, this repo's server, …) **will fail to load on your GTX 1650**. So the demo
> ships **no** pre-built engines — you must run `build_trt.py` **on the GTX 1650 itself**. Never
> copy `.engine` files between different GPUs; rebuild instead.

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
python build_trt.py                                  # everything: 4 tasks x 2 arms x {fp32,fp16}
python build_trt.py --jobs CelebA LFW --precisions fp16   # or just what you need (faster)
```

It prints the GPU it is building on — make sure that is your **GTX 1650**. Classification tasks
export encoder+head (→ logits); LFW exports the encoder (→ 768-d embedding). Workspace is capped
at 1 GB so it fits the 4 GB card.

**3c.** Launch the app (step 4) and pick **TensorRT (FP16)** / **(FP32)** in the backend dropdown.
A model with no engine, or if TensorRT isn't importable, silently falls back to PyTorch.

## 4. Run

```bash
python app.py                       # then open http://127.0.0.1:7860
```
- **Backend dropdown** (top of the page): `PyTorch`, `TensorRT (FP16)`, or `TensorRT (FP32)`.
  TensorRT options use the engines from step 3; if an engine is missing or TensorRT isn't
  installed, that model transparently falls back to PyTorch.
- Force CPU: `DEMO_DEVICE=cpu python app.py` (PowerShell: `$env:DEMO_DEVICE="cpu"; python app.py`).

## Latency & peak VRAM

Both are measured on the **single forward that produced the prediction you are looking at** — the
demo adds no extra passes, so turning the table on costs nothing at inference time.

- **Latency** — CUDA events around that forward (wall-clock `perf_counter` on CPU). CUDA events
  measure the GPU's own execution window rather than the wall time around an asynchronous launch.
  Each model warms up **once** per task and backend before its first timed run: the very first
  forward pays cuDNN autotuning and allocator growth, which would otherwise land entirely on your
  first click and make the Base-vs-Ghost ratio meaningless.
- **Peak VRAM** — that model's **weights + the activation peak of that same forward**, taken as the
  rise in `max_memory_allocated` over the pre-forward baseline. Reporting the raw process-wide peak
  instead — what a naive `max_memory_allocated()` call gives — counts the *other* cached arm and the
  CUDA context, inflating both models to roughly the same ~2 GB and hiding the very difference this
  demo exists to show. On CPU the column reads `— (CPU)`; if your NVIDIA GPU shows that, your
  PyTorch is a CPU-only build.
- **Paper A5000** — the report's **FP16, batch-32** figure from `resource_meta.json` (Base 771 /
  Ghost 609 MB). Batch 32 against batch 1 is why it is larger: use it as the reference regime, and
  use the measured column to compare the two arms on your own machine.

## Notes
- `casia_baseline.pth` is a freshly retrained demo checkpoint; the report's CASIA numbers are unchanged.
- SCface checkpoints: **base = seed1** (Top-1 43.5%), **ghost = seed0** (Top-1 31.7%). *(seed0's base
  file on Drive had been clobbered to an untrained epoch-0 state, which made ConvMAE-Base output
  uniform "random" predictions; `fetch_checkpoints.py` now pins the correct seed1 file and its md5, so
  a previously-downloaded broken copy is re-fetched.)* SCface is the low-res stressor: Base is only
  ~44% Top-1, so its top-1 is often wrong — pick a *surveillance* crop (the tested domain) rather than
  a high-res mugshot, and read the top-5. Predictions are subject IDs `001–130` matching the
  filename's leading number. The app also self-checks each classification checkpoint and shows a
  warning if a loaded head looks untrained, so a bad file never silently degrades to random again.
- Preprocessing matches the fine-tune **eval transform exactly** (Resize 256 **bicubic** → CenterCrop
  224 → ImageNet normalise); using bilinear here previously degraded predictions, worst on SCface.

## Layout
```
demo_app/
  app.py                      Gradio UI + inference (PyTorch / TensorRT backend selector)
  dataset_paths.py            where every split lives; the only place layout is declared
  face_models.py              base/ghost model builders + checkpoint loader
  trt_backend.py              TensorRT engine loader + runtime
  models/                     bundled architecture code (ConvViT, Ghost blocks)

  fetch_checkpoints.py        downloads checkpoints into checkpoints/
  fetch_datasets.py           downloads the original datasets into datasets/ (Kaggle)
  extract_casia_recordio.py   CASIA RecordIO -> prepared/{train,val}/<label>/
  split_celeba.py             CelebA flat folder -> {train,test}/
  calibrate_lfw.py            refit the LFW cosine threshold on the official pairs
  build_trt.py                export ONNX + build TensorRT engines into engines/

  resource_meta.json          reference A5000 resource numbers
  advice/casia_class_maps/    the fine-tune's own CASIA class ordering (+ its prepare script)
  requirements.txt  run.sh  run.bat
  checkpoints/  engines/  datasets/   (populated by the scripts above)
```
