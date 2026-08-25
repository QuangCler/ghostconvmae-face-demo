"""Face-model inference demo (Gradio) — ConvMAE-Base vs Ghost+ConvMAE.

Side-by-side comparison of the two fine-tuned backbones on four face tasks. Each tab describes
what the models do, offers an in-UI image picker (from the task's dataset, or drop your own), and
shows a compact params+latency table. Identity tabs also show the predicted person's face; the
CelebA tab shows the ground-truth attributes when the image belongs to the dataset. Inference runs
through native PyTorch or a TensorRT engine (build_trt.py), selectable in the UI.

Run path is tuned for latency: each model is warmed up once per (task, model, backend), then every
click times the single real forward that produces the prediction — no extra iterations.

Launch:  python app.py        (then open the printed http://127.0.0.1:7860)
"""
import os
import csv
import inspect
import json
import statistics
import time
from collections import deque

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from face_models import build_and_load, embed
import linprobe_models as clsm
import dataset_paths as dsp
import trt_backend

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")
ENGINES = os.path.join(HERE, "engines")
DATASETS = os.path.join(HERE, "datasets")

_forced = os.environ.get("DEMO_DEVICE")
DEVICE = _forced or ("cuda" if torch.cuda.is_available() else "cpu")
CUDA = DEVICE == "cuda" and torch.cuda.is_available()

CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

TOP_K = 5
ID_THRESHOLD = 0.50      # above this the top-1 counts as a confident match
ATTR_THRESHOLD = 0.50    # CelebA is multi-label: show every attribute scoring above this
# LFW verdict cut on L2-normalised cosine. The old 0.50 was far too strict for these embeddings:
# on 200 sampled multi-photo people it kept every non-match out (TNR 100%) but recognised barely
# half the true ones (TPR 50% base / 38% ghost), so genuine pairs read "DIFFERENT". The separation
# was never the problem — ROC-AUC is 0.9925 / 0.9809 — only where the cut sat. Best accuracy on
# that sample was at 0.29 (base) and 0.23 (ghost); 0.235 is the shared value in use.
LFW_THRESHOLD = 0.235
_UNTRAINED_STD = 0.005
_WARMUP = 2
_LAT_WINDOW = 7          # median over this many recent real forwards, to ride out system jitter

JOBS = {
    "CelebA": {
        "task": "attribute", "num_classes": 40, "title": "CelebA — Facial attributes", "dataset": "celeba",
        "ckpts": {"baseline": "celeba_baseline.pth", "ghost": "celeba_ghost.pth"},
        "purpose": ("Multi-label attribute recognition — each backbone scores 40 attributes "
                    "(Smiling, Eyeglasses, Male, …) and the panel lists every one it puts above "
                    "50%, so a busy face shows more than a plain one. When the image belongs to "
                    "CelebA, its ground-truth attributes are listed and each model is scored "
                    "against them."),
        "ref": "Paper (3 seeds): mAP 0.789 (Base) vs 0.778 (Ghost)."},
    "CASIA": {
        "task": "identity", "num_classes": None, "title": "CASIA-WebFace — Identity", "dataset": "casia",
        "ckpts": {"baseline": "casia_baseline.pth", "ghost": "casia_ghost.pth"},
        "purpose": ("Closed-set face identification — each backbone predicts the person; the panel "
                    "shows the top-5 candidates and the predicted person's face."),
        "ref": "Paper (3 seeds): Top-1 91.49% (Base) vs 91.32% (Ghost). Demo uses a lightweight "
               "retrained CASIA checkpoint; report numbers are unchanged.",
        "setup": ("**Input** — `datasets/casia/prepared/val/<label>/`, the held-out split "
                  "(2 images per identity, matching the fine-tune's `val_per_class=2`).\n\n"
                  "**Predicted face** — `datasets/casia/prepared/train/<label>/`, an image the model "
                  "did train on. The class index is named through `datasets/casia/class_order.json`, "
                  "which is `ImageFolder(train_dir).classes` from the fine-tune run itself — no "
                  "reconstruction, no guessing.\n\n"
                  "**Note on the labels** — folder names are the RecordIO integer labels, not CASIA "
                  "person ids, and they are not contiguous: label `09282` has only 2 images, so both "
                  "went to val and it never reached `train/`. That is why the head is 10,571 and "
                  "class index `i` is label `i` below 9282 and `i+1` from 9282 on.")},
    "SCface": {
        "task": "identity", "num_classes": None, "title": "SCface — Cross-resolution identity", "dataset": "scface",
        "ckpts": {"baseline": "scface_baseline.pth", "ghost": "scface_ghost.pth"},
        "purpose": ("Cross-resolution identification (130 subjects) — trained on high-res mugshots, "
                    "tested on low-res surveillance crops; the stress case where Ghost's compression "
                    "costs most. Predictions are subject IDs 001–130, with the predicted mugshot."),
        "ref": "Paper (3 seeds): Top-1 45.13% (Base) vs 31.36% (Ghost) — low-resolution stressor.",
        "setup": ("Read straight from the SCface distribution as it ships — no re-foldering needed.\n\n"
                  "**Input** — `datasets/scface/surveillance_cameras_all/<id>_cam<n>_<distance>.jpg`. "
                  "The fine-tune only ever saw mugshots, so every one of these crops is unseen *and* "
                  "out-of-domain: the strictest of the four tests.\n\n"
                  "**Predicted face** — `datasets/scface/mugshot_frontal_cropped_all/<id>_frontal.JPG`, "
                  "the high-res mugshot the model actually trained on.\n\n"
                  "Base Top-1 is ~44%, so many crops land below the 50% threshold and read as "
                  "'no confident match' — the face is still shown, marked low confidence.")},
    "LFW": {
        "task": "verification", "num_classes": 1680, "title": "LFW — Face verification", "dataset": "lfw",
        "ckpts": {"baseline": "lfw_baseline.pth", "ghost": "lfw_ghost.pth"},
        "purpose": ("Verification — each backbone maps a face to a 768-d embedding; we report the "
                    f"cosine similarity between the two faces and call it the same person above "
                    f"{LFW_THRESHOLD}. Open-set: no identity here is a training class."),
        "ref": "Paper (3 seeds): ROC-AUC 0.9921 (Base) vs 0.9833 (Ghost)."},
    "ImageNet": {
        "task": "classification", "num_classes": 1000,
        "title": "ImageNet-1K — Linear probe (90-ep head on the 300-ep pretrain)",
        "dataset": "imagenet",
        "ckpts": {"baseline": "imagenet_baseline.pth", "ghost": "imagenet_ghost.pth"},
        "purpose": ("1000-way ImageNet classification with the backbone **frozen** — only a linear "
                    "head (BatchNorm→Linear) was trained for 90 epochs on top of each 300-epoch "
                    "pretrained backbone. This is the representation-quality probe: it reads the "
                    "features as they were pretrained, without fine-tuning. The panel shows each "
                    "backbone's top-5 classes for the chosen image."),
        "ref": "ImageNet-1K linear-probe Top-1 (report): 64.06% (Base) vs 58.60% (Ghost).",
        "setup": ("**Input** — the standard ImageNet-1K **validation** set (50,000 held-out images), "
                  "which the probe never trained on. Get it with `python fetch_datasets.py --only "
                  "imagenet` (the Kaggle `imagenet-object-localization-challenge`, val ≈ 6.4 GB), or "
                  "point `IMAGENET_VAL_DIR` at a copy you already have. Class names come from the "
                  "bundled `imagenet_class_index.json`.\n\n"
                  "**Only ConvMAE-Base and Ghost+ConvMAE are shown here — the two Mamba arms "
                  "(Ghost+ForwardMamba, Ghost+BiMamba) are deliberately excluded from this demo.** "
                  "Their Stage-3 Mamba blocks need the CUDA-only `mamba_ssm` selective-scan kernels, "
                  "which do not install on the target laptop GPU (a GTX 1650), and the selective-scan "
                  "op has no ONNX/TensorRT path — so those arms can neither run in PyTorch here nor be "
                  "exported to a TensorRT engine. They are covered in `LINPROBE_DEPLOY.md` for a "
                  "CUDA server instead.")},
}

BACKENDS = {"PyTorch": ("pytorch", None), "TensorRT (FP16)": ("trt", "fp16"), "TensorRT (FP32)": ("trt", "fp32")}

try:
    _BICUBIC = transforms.InterpolationMode.BICUBIC
except AttributeError:
    import PIL
    _BICUBIC = PIL.Image.BICUBIC
PREPROC = transforms.Compose([
    transforms.Resize(256, interpolation=_BICUBIC), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

_CACHE = {}
_WARMED = set()
_LATENCY = {}            # (job, arm, backend, feats, label) -> recent forward times, for the median
_CELEBA_TRUTH = None     # {basename: [true attribute names]}
try:
    with open(os.path.join(HERE, "resource_meta.json")) as _f:
        _REF = json.load(_f)          # the report's A5000 reference numbers
except Exception:
    _REF = {}
try:
    with open(os.path.join(HERE, "lfw_thresholds.json")) as _f:
        _LFW_THR = json.load(_f)      # per-arm cosine cut, from calibrate_lfw.py
except Exception:
    _LFW_THR = {}
ARM_NAME = {"baseline": "ConvMAE-Base", "ghost": "Ghost+ConvMAE"}


# ---------------------------------------------------------------- dataset access
# Layout, splits and class ordering all live in dataset_paths.py. The rule the UI enforces:
# every picker offers HELD-OUT images only, and the identity tabs read the predicted person's
# face from the TRAIN split, so an unseen input is scored against a face the model did learn.
def celeba_truth():
    global _CELEBA_TRUTH
    if _CELEBA_TRUTH is not None:
        return _CELEBA_TRUTH
    _CELEBA_TRUTH = {}
    path = dsp.celeba_attr_csv()
    if not path:
        return _CELEBA_TRUTH
    try:
        with open(path, newline="") as f:
            r = csv.reader(f)
            cols = {name: i for i, name in enumerate(next(r))}
            attr_cols = [(a, cols[a]) for a in CELEBA_ATTRS if a in cols]
            idcol = cols.get("image_id", 0)
            for row in r:
                if row:
                    _CELEBA_TRUTH[os.path.basename(row[idcol])] = [a for a, i in attr_cols if row[i] == "1"]
    except Exception:
        pass
    return _CELEBA_TRUTH


def load_sample(job, rel):
    """Returns (image path, caption, rel) — rel is tracked so CelebA can look up ground truth."""
    if not rel:
        return None, "", None
    full = dsp.test_samples(job).get(rel)
    if not full or not os.path.exists(full):
        return None, "Sample not found — re-run `python fetch_datasets.py`.", None
    return full, f"Held-out · `{rel}`", rel


def _open(path):
    return Image.open(path).convert("RGB") if path else None


def dropped_celeba(path):
    """Caption for a dropped CelebA-tab image, and the name to score it against."""
    if not path:
        return None, ""
    name = os.path.basename(path)
    part = dsp.celeba_partition(name)
    if part is None:
        return None, f"Dropped · `{name}` — not a CelebA image, so no ground truth to score against."
    where = {"train": "**partition 0 (train)** — the model was fitted on this image, so treat the "
                      "score as memorisation, not generalisation",
             "val": "**partition 1 (val)** — held out",
             "test": "**partition 2 (test)** — held out"}[part]
    return name, f"Dropped · `{name}` — a CelebA image from {where}."


def dropped_identity(job, path):
    """Caption for a dropped identity-tab image: which split it belongs to, if any."""
    if not path:
        return ""
    name = os.path.basename(path)
    hit = dsp.locate(job, name)
    if not hit:
        return f"Dropped · `{name}` — not in the {JOBS[job]['dataset']} dataset, so there is no " \
               "true identity to compare the prediction against."
    split, identity = hit
    label = _fmt_id(job, None, identity)
    if split == "train":
        return (f"Dropped · `{name}` — **train** image of {label}. The model was fitted on this "
                "exact picture, so a correct answer here is memorisation.")
    return f"Dropped · `{name}` — **held-out** image of {label}. True answer: {label}."


# ---------------------------------------------------------------- models
def _head_size(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    return sd["head.weight"].shape[0]


def get_model(job, arm):
    key = (job, arm)
    if key in _CACHE:
        return _CACHE[key]
    spec = JOBS[job]
    path = os.path.join(CKPT, spec["ckpts"][arm])
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    nc = spec["num_classes"] or _head_size(path)
    if spec["task"] == "classification":
        model = clsm.build_and_load_cls(arm, nc, path, map_location=DEVICE).to(DEVICE).eval()
    else:
        model = build_and_load(arm, nc, path, map_location=DEVICE).to(DEVICE).eval()
    try:
        head_std = float(model.head.weight.detach().float().std())
    except Exception:
        head_std = None
    weights_B = (sum(p.numel() * p.element_size() for p in model.parameters())
                 + sum(b.numel() * b.element_size() for b in model.buffers()))
    meta = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6, "n_classes": nc,
            "weights_MB": weights_B / 1e6, "ckpt_MB": os.path.getsize(path) / 1e6,
            "head_std": head_std, "untrained": head_std is not None and head_std < _UNTRAINED_STD}
    _CACHE[key] = (model, meta)
    return _CACHE[key]


@torch.no_grad()
def infer_one(job, arm, x, backend, feats=False):
    model, meta = get_model(job, arm)
    kind, prec = BACKENDS.get(backend, ("pytorch", None))
    label, call = "PyTorch", None
    if kind == "trt" and CUDA and trt_backend.available():
        epath = trt_backend.engine_path(ENGINES, job, arm, prec, feats)
        if os.path.exists(epath):
            label, call = f"TensorRT-{prec.upper()}", (lambda z: trt_backend.infer(epath, z))
        else:
            label = "PyTorch (no engine)"
    torch_call = (lambda z: embed(model, z)) if feats else (lambda z: model(z))
    if call is None:
        call = torch_call

    def timed(fn):
        """Time the one real forward and capture its activation peak — no extra passes.

        Latency uses CUDA events, which measure the GPU's own execution window rather than wall
        clock around an async launch. The VRAM figure is this model's activation peak (the
        `max_memory_allocated` rise over the pre-forward baseline), which the caller adds to the
        model's own weights. Taking the raw process-wide peak instead would count the *other*
        cached arm and the CUDA context, inflating both models to the same ~2 GB and hiding the
        very difference the demo exists to show.
        """
        if CUDA:
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            s.record(); o = fn(x); e.record()
            torch.cuda.synchronize()
            act = max(0, torch.cuda.max_memory_allocated() - base) / 1e6
            return o, s.elapsed_time(e), act
        t0 = time.perf_counter(); o = fn(x)
        return o, (time.perf_counter() - t0) * 1000, None

    # Warm up once per (task, arm, backend, path): the first forward pays cuDNN autotuning and
    # allocator growth, which would otherwise land entirely on the user's first click and make the
    # Base-vs-Ghost ratio meaningless. Steady-state clicks time a warmed forward only.
    wkey = (job, arm, backend, feats)
    if wkey not in _WARMED:
        for _ in range(_WARMUP):
            call(x)
        _WARMED.add(wkey)
    try:
        out, ms, act = timed(call)
    except Exception as e:
        label, (out, ms, act) = f"PyTorch (TRT failed: {type(e).__name__})", timed(torch_call)

    # Report the median of the recent real forwards rather than the last one. A single batch-1
    # forward on a laptop GPU swings 38-77 ms with background load, which swamps the ~10% gap
    # between the two arms; the median of the last few settles without running anything extra.
    hist = _LATENCY.setdefault((job, arm, backend, feats, label), deque(maxlen=_LAT_WINDOW))
    hist.append(ms)
    return out, statistics.median(hist), label, dict(meta, act_MB=act, samples=len(hist))


# ---------------------------------------------------------------- prediction formatting
def _fmt_id(job, c, name):
    if name is not None:
        return f"subject {name}" if job == "SCface" else f"ID {name}"
    return f"subject {int(c) + 1:03d}" if job == "SCface" else f"identity #{int(c)}"


def _attr_dict(out):
    """Every attribute the model asserts, i.e. sigmoid > ATTR_THRESHOLD, most confident first.

    Attribute recognition is multi-label: how many attributes are true varies per face, so a fixed
    top-5 both hid real positives on a busy face and padded a plain one with sub-50% guesses. When
    nothing clears the bar the top-3 are shown so the panel still says what the model leaned toward.
    """
    probs = torch.sigmoid(out)[0].float().cpu().numpy()
    order = np.argsort(-probs)
    hits = [i for i in order if probs[i] > ATTR_THRESHOLD]
    return {CELEBA_ATTRS[i]: float(probs[i]) for i in (hits or order[:3])}


def _identity(job, out, meta):
    """Return (top-5 label dict, top-1 face list for gr.Gallery, status markdown).

    The face comes from the TRAIN split (CASIA: an image past the held-out ones; SCface: the
    subject's mugshot), so it answers "who does the model think this is" with something the model
    actually learned. It is shown whenever it resolves — the 50% threshold decides the *verdict*
    wording, not whether the picture appears, so a low-confidence guess still shows its face.
    """
    probs = out.float().softmax(-1)[0].cpu()
    v, idx = probs.topk(min(TOP_K, probs.numel()))
    d = {}
    for p, c in zip(v.tolist(), idx.tolist()):
        d[_fmt_id(job, c, dsp.class_label(job, c))] = float(p)
    top_p, top_c = float(v[0]), int(idx[0])
    lab = _fmt_id(job, top_c, dsp.class_label(job, top_c))
    confident = top_p > ID_THRESHOLD
    face = dsp.class_face(job, top_c)
    if confident:
        msg = f"Predicted **{lab}** · {top_p * 100:.0f}% confidence."
    else:
        msg = f"**No confident match** — best guess {lab} at {top_p * 100:.0f}% (threshold 50%)."
    if face:
        tag = "" if confident else " · low confidence"
        return d, [(face, f"{lab} · {top_p * 100:.0f}%{tag}")], msg
    if job == "CASIA" and not dsp.casia_split("train"):
        return d, [], (f"{msg}\n\n_No identity folders on disk — run "
                       "`python extract_casia_recordio.py`._")
    return d, [], f"{msg} No train image on disk for this class — see *Where the faces come from* below."


def _fit_score(pred_probs, truth):
    """How well one model's 40 attribute calls match the CelebA annotation.

    Two numbers, because they answer different questions: **accuracy** over all 40 binary calls is
    the headline but is flattered by the many attributes that are negative for almost every face,
    so **F1 on the positives** is reported next to it.
    """
    truth_set = set(truth)
    pred_set = {a for a, p in zip(CELEBA_ATTRS, pred_probs) if p > ATTR_THRESHOLD}
    correct = sum(1 for a in CELEBA_ATTRS if (a in pred_set) == (a in truth_set))
    tp = len(pred_set & truth_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(truth_set) if truth_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return correct / len(CELEBA_ATTRS), f1, correct


RES_HEADERS = ["Model", "Backend", "Params (M)", "Weights (MB)", "Latency (ms)",
               "Peak VRAM (MB)", "Paper A5000 (MB)"]


def _table(rows):
    """One row per arm plus a Ghost-vs-Base delta row.

    Peak VRAM is this model's weights plus its own activation peak at batch 1 — comparable between
    the arms and stable across clicks. The paper column is the report's A5000 / FP16 / batch-32
    figure, a deliberately different regime: read it as the reference, not as a target.
    """
    t, got = [], {}
    for arm, meta, ms, label in rows:
        name = ARM_NAME.get(arm, arm)
        ref = _REF.get(arm, {}).get("vram_fp16_MB", "—")
        if meta is None:
            t.append([name, "checkpoint missing", "—", "—", "—", "—", str(ref)])
            continue
        act = meta.get("act_MB")
        vram = f"{meta['weights_MB'] + act:.0f}" if act is not None else "— (CPU)"
        t.append([name, label, f"{meta['params_M']:.1f}", f"{meta['weights_MB']:.0f}",
                  f"{ms:.1f}", vram, str(ref)])
        got[arm] = (meta["params_M"], ms, (meta["weights_MB"] + act) if act is not None else None)
    if "baseline" in got and "ghost" in got:
        b, g = got["baseline"], got["ghost"]

        def delta(x, y):
            return f"{100 * (y - x) / x:+.0f}%" if x and y else "—"

        t.append(["Ghost vs Base (Δ)", "", delta(b[0], g[0]), "", delta(b[1], g[1]),
                  delta(b[2], g[2]), delta(_REF.get("baseline", {}).get("vram_fp16_MB"),
                                           _REF.get("ghost", {}).get("vram_fp16_MB"))])
    return t


RES_NOTE = ("**Latency** is the median of your recent real forwards for this model and backend "
            "(CUDA events, batch 1) — no extra passes are run, and the median rides out the system "
            "jitter that makes any single batch-1 timing swing wildly. Each model warms up once per "
            "task and backend, so the first click pays the cuDNN autotune. **Peak VRAM** is that model's weights "
            "plus the activation peak of that same forward, so it excludes the other cached arm and "
            "the CUDA context and stays comparable between the two. **Paper A5000** is the report's "
            "FP16 batch-32 figure — a different regime, larger by design.")


def _warn(rows):
    out = [f"⚠ **{ARM_NAME[a]}**'s checkpoint looks untrained (head σ≈{m['head_std']:.4f}); predictions "
           f"are ~random. Re-run `python fetch_checkpoints.py` for the corrected file."
           for a, m, *_ in rows if m and m.get("untrained")]
    out += [f"⚠ **{ARM_NAME[a]}**'s checkpoint is missing — run `python fetch_checkpoints.py`."
            for a, m, *_ in rows if m is None]
    return out


# ---------------------------------------------------------------- run
def run_attribute(image, backend, sel_rel):
    if image is None:
        return {}, {}, "", [], "Choose a sample or drop an image, then run."
    x = PREPROC(_open(image)).unsqueeze(0).to(DEVICE)
    labels, probs, rows = {"baseline": {}, "ghost": {}}, {}, []
    for arm in ("baseline", "ghost"):
        try:
            out, ms, label, meta = infer_one("CelebA", arm, x, backend)
        except FileNotFoundError:
            rows.append((arm, None, None, None)); continue
        rows.append((arm, meta, ms, label))
        probs[arm] = torch.sigmoid(out)[0].float().cpu().numpy()
        labels[arm] = _attr_dict(out)

    truth = celeba_truth().get(os.path.basename(sel_rel)) if sel_rel else None
    if truth is None:
        gt = "_No CelebA annotation for this image, so there is nothing to score against._"
    else:
        part = dsp.celeba_partition(sel_rel)
        head = "**Ground truth** — " + (", ".join(truth) if truth else "_(none marked positive)_")
        if part == "train":
            head += "\n\n⚠ This image is in **partition 0 (train)** — the scores below are memorisation."
        lines = [head, ""]
        for arm in ("baseline", "ghost"):
            if arm not in probs:
                continue
            acc, f1, correct = _fit_score(probs[arm], truth)
            lines.append(f"- **{ARM_NAME[arm]}** — fits ground truth **{acc * 100:.1f}%** "
                         f"({correct}/{len(CELEBA_ATTRS)} attributes) · F1 on positives `{f1:.2f}`")
        gt = "\n".join(lines)
    note = (("\n\n".join(_warn(rows)) + "\n\n") if _warn(rows) else "") + RES_NOTE
    return labels["baseline"], labels["ghost"], gt, _table(rows), note


def run_identity(job, image, backend):
    if image is None:
        return {}, [], "", {}, [], "", [], "Choose a sample or drop an image, then run."
    x = PREPROC(_open(image)).unsqueeze(0).to(DEVICE)
    res, rows = {}, []
    for arm in ("baseline", "ghost"):
        try:
            out, ms, label, meta = infer_one(job, arm, x, backend)
        except FileNotFoundError:
            res[arm] = ({}, [], "_Checkpoint missing._")
            rows.append((arm, None, None, None)); continue
        rows.append((arm, meta, ms, label))
        res[arm] = _identity(job, out, meta)
    lb, gb, mb = res.get("baseline", ({}, [], ""))
    lg, gg, mg = res.get("ghost", ({}, [], ""))
    note = (("\n\n".join(_warn(rows)) + "\n\n") if _warn(rows) else "") + RES_NOTE
    return lb, gb, mb, lg, gg, mg, _table(rows), note


def _classification(out):
    """Top-5 ImageNet classes for one backbone: {class name: probability}, most confident first."""
    probs = out.float().softmax(-1)[0].cpu()
    v, idx = probs.topk(min(TOP_K, probs.numel()))
    d = {}
    for p, c in zip(v.tolist(), idx.tolist()):
        d[dsp.class_label("ImageNet", int(c)) or f"class #{int(c)}"] = float(p)
    return d


def run_classification(image, backend):
    if image is None:
        return {}, {}, [], "Choose a sample or drop an image, then run."
    x = PREPROC(_open(image)).unsqueeze(0).to(DEVICE)
    labels, rows = {"baseline": {}, "ghost": {}}, []
    for arm in ("baseline", "ghost"):
        try:
            out, ms, label, meta = infer_one("ImageNet", arm, x, backend)
        except FileNotFoundError:
            rows.append((arm, None, None, None)); continue
        rows.append((arm, meta, ms, label))
        labels[arm] = _classification(out)
    note = (("\n\n".join(_warn(rows)) + "\n\n") if _warn(rows) else "") + RES_NOTE
    return labels["baseline"], labels["ghost"], _table(rows), note


def run_verify(image_a, image_b, backend):
    if image_a is None or image_b is None:
        return "Choose an image for A and B (person → photo), then verify.", [], ""
    xa = PREPROC(image_a.convert("RGB")).unsqueeze(0).to(DEVICE)
    xb = PREPROC(image_b.convert("RGB")).unsqueeze(0).to(DEVICE)
    # Both faces go through as one batch-2 forward per arm — half the GPU work, and half the
    # warm-up, which is what made the first Verify click slow. TRT engines are built for batch 1,
    # so that path still runs the two images separately.
    trt = BACKENDS.get(backend, ("pytorch", None))[0] == "trt" and CUDA and trt_backend.available()
    rows, cards = [], []
    for arm in ("baseline", "ghost"):
        try:
            if trt:
                ea, ma, label, meta = infer_one("LFW", arm, xa, backend, feats=True)
                eb, mb, _, _ = infer_one("LFW", arm, xb, backend, feats=True)
                ms = (ma + mb) / 2
            else:
                out, ms, label, meta = infer_one("LFW", arm, torch.cat([xa, xb]), backend, feats=True)
                ea, eb = out[:1], out[1:]
        except FileNotFoundError:
            rows.append((arm, None, None, None)); continue
        ea = torch.nn.functional.normalize(ea.float(), dim=-1)
        eb = torch.nn.functional.normalize(eb.float(), dim=-1)
        cos = float((ea * eb).sum().cpu())
        thr = _LFW_THR.get(arm, {}).get("threshold", LFW_THRESHOLD)
        verdict = "SAME person" if cos > thr else "DIFFERENT person"
        cards.append(f"- **{ARM_NAME[arm]}** — cosine `{cos:+.3f}` vs threshold `{thr:.3f}` "
                     f"→ **{verdict}**")
        rows.append((arm, meta, ms, label))
    foot = (f"\n\n_Verdict threshold: cosine > {LFW_THRESHOLD} on L2-normalised embeddings. "
            "Run `python calibrate_lfw.py` to refit it per arm on the official pairs._")
    body = ("\n".join(cards) if cards else "_No checkpoints bundled._") + foot
    note = (("\n\n".join(_warn(rows)) + "\n\n") if _warn(rows) else "") + \
        RES_NOTE + "\n\nBoth faces go through as one batch-2 forward, so the latency covers the pair."
    return body, _table(rows), note


# ---------------------------------------------------------------- UI
# Runs once on load: force dark mode (Gradio otherwise follows the OS), then wire the dropdowns
# to page themselves. Gradio exposes no scroll event on a Dropdown, so this attaches one to the
# option list the component renders when opened and clicks that picker's "load more" button on
# reaching the bottom. If a Gradio version changes that markup the button is still there to click.
FORCE_DARK = """
() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme', 'dark'); window.location.replace(u.toString()); return;
  }
  // The option list is `position: fixed`, so it is not reliably a DOM descendant of the column
  // holding its "load more" button — walking up from it found nothing. Only one tab is visible at
  // a time and each has exactly one picker, so the visible .morerow button is unambiguous.
  const visibleMoreButton = () => {
    for (const btn of document.querySelectorAll('.morerow button')) {
      if (btn.offsetParent !== null && !btn.disabled) return btn;
    }
    return null;
  };
  let lastClick = 0;
  const maybeLoad = (list) => {
    if (list.scrollTop + list.clientHeight < list.scrollHeight - 64) return;
    const now = Date.now();
    if (now - lastClick < 700) return;          // one page per scroll-to-bottom, not per event
    const btn = visibleMoreButton();
    if (btn) { lastClick = now; btn.click(); }
  };
  const attach = (list) => {
    if (list.dataset.autopage) return;
    list.dataset.autopage = '1';
    list.addEventListener('scroll', () => maybeLoad(list), {passive: true});
    list.addEventListener('wheel', () => setTimeout(() => maybeLoad(list), 60), {passive: true});
  };
  new MutationObserver(() => {
    document.querySelectorAll('ul.options, ul[role="listbox"]').forEach(attach);
  }).observe(document.body, {childList: true, subtree: true});
}
"""

CSS = """
/* --- soft dark 'console' palette; override theme variables in BOTH modes --- */
.gradio-container, .gradio-container.dark, .dark, :root {
  --body-text-color:#dfe7ea !important; --body-text-color-subdued:#93a3ab !important;
  --body-background-fill:transparent !important; --background-fill-primary:#141d23 !important;
  --background-fill-secondary:#0f171c !important;
  --block-background-fill:#18232a !important;
  --block-label-background-fill:#12272b !important; --block-label-text-color:#5fd6dc !important;
  --block-title-text-color:#c6d2d8 !important; --block-border-color:#27353d !important;
  --border-color-primary:#27353d !important; --input-background-fill:#0f181d !important;
  --input-border-color:#2c3b43 !important; --input-placeholder-color:#6b7b83 !important;
  --input-text-color:#dfe7ea !important; --block-info-text-color:#8a9aa2 !important;
  --table-even-background-fill:#141e24 !important; --table-odd-background-fill:#18232a !important;
  --table-border-color:#27353d !important; --neutral-950:#dfe7ea !important; --neutral-50:#0f171c !important;
}
.gradio-container {background:
  radial-gradient(1100px 380px at 78% -8%, rgba(43,192,199,.10), transparent 60%),
  linear-gradient(180deg,#0e151a 0%,#121a20 100%) !important; color:#dfe7ea !important;
  font-family:'IBM Plex Sans', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  max-width:1140px !important; margin:0 auto !important; padding:14px 12px 26px !important;}
.gradio-container, .gradio-container p, .gradio-container span, .gradio-container div,
.gradio-container li, .gradio-container td, .gradio-container th, .gradio-container label {color:#dfe7ea;}

#hero {display:flex; justify-content:space-between; align-items:center; gap:18px;
  background:linear-gradient(120deg,#122a31 0%,#16323b 55%,#173a44 130%);
  border:1px solid #244049; border-radius:16px; padding:19px 23px;
  box-shadow:0 8px 26px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.04); margin-bottom:14px;}
#hero h1 {font-size:21px; font-weight:700; margin:0 0 3px; letter-spacing:-.2px; color:#f0f6f7 !important;}
#hero p {margin:0; font-size:12.5px; color:#a9bcc2 !important; line-height:1.55; max-width:74ch;}
.statuspill {white-space:nowrap; font-family:'IBM Plex Mono', ui-monospace, monospace;
  font-size:11.5px; font-weight:600; letter-spacing:.02em; padding:8px 12px; border-radius:9px; border:1px solid #2c3b43;}
.statuspill.on {background:rgba(45,190,130,.14); border-color:rgba(45,190,130,.4);} .statuspill.on, .statuspill.on * {color:#57d99a !important;}
.statuspill.off {background:#1a2830;} .statuspill.off, .statuspill.off * {color:#93a3ab !important;}

.purpose {background:linear-gradient(180deg,#152329,#121e24); border:1px solid #27353d; border-left:3px solid #2bc0c7;
  border-radius:12px; padding:12px 15px; margin:6px 0 12px; font-size:13px; line-height:1.6;
  box-shadow:0 2px 10px rgba(0,0,0,.2);}
.purpose *, .purpose b {color:#cdd9de !important;} .purpose i {color:#5fd6dc !important; font-style:normal; font-weight:600;}

.card {background:linear-gradient(180deg,#1b262d,#161f25) !important; border:1px solid #27353d !important;
  border-radius:16px !important; padding:16px !important;
  box-shadow:0 6px 20px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.03) !important;}
.selname, .selname * {font-family:'IBM Plex Mono', ui-monospace, monospace; color:#5fd6dc !important; font-size:12px; font-weight:600;}
.note, .note * {color:#8a9aa2 !important; font-size:11.5px; line-height:1.6;}
.idmsg, .idmsg * {color:#c6d2d8 !important; font-size:12.5px; line-height:1.55;}
.truth, .truth * {color:#c6d2d8 !important; font-size:12.5px; line-height:1.6;}

.gradio-container label, .gradio-container .label-wrap span, .gradio-container legend,
.gradio-container .block-title {color:#c6d2d8 !important; font-weight:600 !important;}
.arm-base .label-wrap {border-left:3px solid #7aa0c4 !important; padding-left:9px !important;}
.arm-ghost .label-wrap {border-left:3px solid #2bc0c7 !important; padding-left:9px !important;}

.gradio-container .tab-nav {border-bottom:1px solid #27353d !important;}
.gradio-container .tab-nav button {color:#93a3ab !important; font-weight:600 !important; background:transparent !important; border:none !important; padding:10px 15px !important;}
.gradio-container .tab-nav button.selected {color:#5fd6dc !important; border-bottom:2.5px solid #2bc0c7 !important;}

button.primary, .gr-button-primary, button.lg.primary {
  background:linear-gradient(180deg,#25b3ba,#189aa1) !important; border:1px solid #2bc0c7 !important;
  color:#04231f !important; font-weight:700 !important; border-radius:11px !important;
  box-shadow:0 6px 18px rgba(43,192,199,.28), inset 0 1px 0 rgba(255,255,255,.25) !important;}
button.primary *, .gr-button-primary * {color:#04231f !important;}
button.primary:hover {filter:brightness(1.07);}
/* Run button sits directly above the results table it fills — compact and centred. */
.runrow {flex:0 0 auto !important; margin:12px 0 8px !important;
  justify-content:center !important; align-items:center !important;}
.runrow button {font-size:13px !important; padding:8px 18px !important; border-radius:9px !important;
  flex-grow:0 !important; width:auto !important;}
/* Keep the option list short enough that it always scrolls — reaching its end is what pages in
   the next batch, so a list tall enough to show every loaded entry would never trigger. */
ul.options {max-height:340px !important;}
.morerow {flex:0 0 auto !important; margin:2px 0 0 !important;}
.morerow button {font-size:11.5px !important; padding:5px 12px !important; border-radius:8px !important;
  flex-grow:0 !important; width:auto !important; background:#17242b !important;
  border:1px solid #2c3b43 !important; color:#8fb9bd !important; font-weight:600 !important;}
.morerow button:hover {border-color:#2bc0c7 !important; color:#5fd6dc !important;}

.gr-dataframe table, table {font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:12.5px;}
.gr-dataframe th, table th {background:#12272b !important; color:#bcd3d6 !important; font-weight:700 !important;
  text-transform:uppercase; letter-spacing:.03em; font-size:11px;}
.gr-dataframe td, table td {color:#dfe7ea !important;}
.gr-dataframe tbody tr:nth-child(even) td {background:#141e24 !important;}
/* Predicted-person thumbnail: mugshots are far larger than the frame, so scale the whole face
   down to fit rather than letting the grid crop it. */
.idgal {border:none !important; box-shadow:none !important; padding:2px 0 0 !important;}
.idgal .grid-container, .idgal .grid-wrap {height:auto !important; overflow:visible !important;}
.idgal img, .idgal .thumbnail-lg img, .idgal button img {
  object-fit:contain !important; width:100% !important; height:100% !important;
  max-height:150px !important; background:#101a1f !important; border-radius:8px !important;}
.idgal .thumbnail-item, .idgal .thumbnail-lg, .idgal button {
  height:auto !important; aspect-ratio:auto !important; background:transparent !important;
  border:1px solid #27353d !important; border-radius:9px !important;}
.idgal .caption, .idgal figcaption {font-family:'IBM Plex Mono', ui-monospace, monospace !important; font-size:10.5px !important; color:#c6d2d8 !important;}
footer {display:none !important;}
"""


def _hero_html():
    if CUDA:
        name = torch.cuda.get_device_name(0)
        trt = "TensorRT ready" if trt_backend.available() else "PyTorch only"
        pill = f"<span class='statuspill on'>● GPU · {name} · {trt}</span>"
    else:
        pill = "<span class='statuspill off'>● CPU · install a CUDA build to use your GPU</span>"
    return ("<div id='hero'><div><h1>Face Model Inference</h1>"
            "<p>ConvMAE-Base vs Ghost+ConvMAE, compared side by side on four face tasks. "
            "Pick a sample or drop a face, choose a backend, and read each model's prediction.</p></div>"
            f"{pill}</div>")


def _style_kwargs(gr):
    """`theme`/`css`/`js` moved from the Blocks constructor to launch() in Gradio 6.

    Returns (blocks_kwargs, launch_kwargs) so the same file styles correctly on 4/5 and on 6 —
    passing them to the wrong one is only a UserWarning, and the app then renders unstyled with
    the dark-mode script never running.
    """
    style = {
        "theme": gr.themes.Soft(primary_hue=gr.themes.colors.teal, neutral_hue=gr.themes.colors.slate,
                                font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
                                font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"]),
        "css": CSS,
        "js": FORCE_DARK,
    }
    if "theme" in inspect.signature(gr.Blocks.launch).parameters:
        return {}, style
    return style, {}


def build_ui():
    import gradio as gr
    blocks_kw, _ = _style_kwargs(gr)
    clearables = []

    with gr.Blocks(title="Face Model Inference — ConvMAE-Base vs Ghost", **blocks_kw) as demo:
        gr.HTML(_hero_html())
        backend = gr.Radio(list(BACKENDS.keys()), value="PyTorch", label="Inference backend",
                           info="TensorRT needs engines built on THIS GPU (build_trt.py); "
                                "missing engine / no TensorRT falls back to PyTorch.")

        def purpose_html(spec):
            return (f"<div class='purpose'><b>{spec['title']}.</b> {spec['purpose']}<br>"
                    f"<i>{spec['ref']}</i></div>")

        def run_button(text="Run both models"):
            """Compact primary button, centred above the results table it fills."""
            with gr.Row(elem_classes="runrow"):
                return gr.Button(text, variant="primary", size="sm", scale=0, min_width=190)

        def sample_picker(job, label):
            """Held-out picker that pages: PAGE entries, another PAGE on scrolling to the end.

            The whole index is built once (it is just paths); only the dropdown's visible choice
            list grows. Loading all of it up front instead pushed the served page from 3 MB to
            24 MB, since every choice is embedded in the payload.
            """
            labels = list(dsp.test_samples(job))
            total = len(labels)

            def more_text(n):
                return f"Load {dsp.PAGE} more  ·  {n}/{total} shown"

            dd = gr.Dropdown(choices=labels[:dsp.PAGE], value=None, label=label,
                             filterable=True, info=dsp.status(job))
            shown = gr.State(min(dsp.PAGE, total))
            with gr.Row(elem_classes="morerow"):
                more = gr.Button(more_text(min(dsp.PAGE, total)), size="sm", scale=0,
                                 min_width=210, visible=total > dsp.PAGE)

            def load_more(n):
                n2 = min(n + dsp.PAGE, total)
                return gr.update(choices=labels[:n2]), n2, gr.update(
                    value=more_text(n2), visible=n2 < total)

            more.click(load_more, inputs=[shown], outputs=[dd, shown, more])
            return dd, shown, more, more_text

        # -------- CelebA (attributes + ground truth) --------
        with gr.Tab("CelebA · Attributes") as t1:
            spec = JOBS["CelebA"]
            gr.HTML(purpose_html(spec))
            sel_state = gr.State(None)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes="card"):
                    dd, shown, more, _ = sample_picker("CelebA", "Held-out CelebA sample")
                    img = gr.Image(type="filepath", label="Input face (or drop your own)", height=300)
                    sel = gr.Markdown("", elem_classes="selname")
                with gr.Column(scale=1, elem_classes="card"):
                    lb = gr.Label(num_top_classes=len(CELEBA_ATTRS), label=ARM_NAME["baseline"],
                                  elem_classes="arm arm-base")
                    lg = gr.Label(num_top_classes=len(CELEBA_ATTRS), label=ARM_NAME["ghost"],
                                  elem_classes="arm arm-ghost")
                    gt = gr.Markdown("", elem_classes="truth")
            run = run_button()
            res = gr.Dataframe(headers=RES_HEADERS, datatype="str", interactive=False,
                               row_count=(3, "fixed"), label="Resource cost", elem_classes="card")
            note = gr.Markdown("", elem_classes="note")
            dd.change(lambda rel: load_sample("CelebA", rel), inputs=[dd], outputs=[img, sel, sel_state])
            img.upload(dropped_celeba, inputs=[img], outputs=[sel_state, sel])
            run.click(run_attribute, inputs=[img, backend, sel_state], outputs=[lb, lg, gt, res, note])
            clearables.extend([(dd, None), (img, None), (sel, ""), (sel_state, None),
                               (lb, None), (lg, None), (gt, ""), (res, []), (note, "")])

        # -------- CASIA / SCface (identity + predicted face) --------
        def identity_tab(job_key):
            spec = JOBS[job_key]
            gr.HTML(purpose_html(spec))
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes="card"):
                    dd, shown, more, _ = sample_picker(job_key, f"Held-out {spec['dataset']} sample")
                    img = gr.Image(type="filepath", label="Input face (or drop your own)", height=300)
                    sel = gr.Markdown("", elem_classes="selname")
                    with gr.Accordion("Where the faces come from", open=False):
                        gr.Markdown(spec["setup"], elem_classes="idmsg")
                with gr.Column(scale=1, elem_classes="card"):
                    lb = gr.Label(num_top_classes=TOP_K, label=ARM_NAME["baseline"], elem_classes="arm arm-base")
                    gb = gr.Gallery(label="Predicted person (train image)", columns=1, height=170,
                                    object_fit="contain", show_label=True, preview=False, elem_classes="idgal")
                    mb = gr.Markdown("", elem_classes="idmsg")
                    lg = gr.Label(num_top_classes=TOP_K, label=ARM_NAME["ghost"], elem_classes="arm arm-ghost")
                    gg = gr.Gallery(label="Predicted person (train image)", columns=1, height=170,
                                    object_fit="contain", show_label=True, preview=False, elem_classes="idgal")
                    mg = gr.Markdown("", elem_classes="idmsg")
            run = run_button()
            res = gr.Dataframe(headers=RES_HEADERS, datatype="str", interactive=False,
                               row_count=(3, "fixed"), label="Resource cost", elem_classes="card")
            note = gr.Markdown("", elem_classes="note")
            dd.change(lambda rel, jk=job_key: load_sample(jk, rel)[:2], inputs=[dd], outputs=[img, sel])
            img.upload(lambda p, jk=job_key: dropped_identity(jk, p), inputs=[img], outputs=[sel])
            run.click(lambda im, bk, jk=job_key: run_identity(jk, im, bk),
                      inputs=[img, backend], outputs=[lb, gb, mb, lg, gg, mg, res, note])
            clearables.extend([(dd, None), (img, None), (sel, ""), (lb, None), (gb, []), (mb, ""),
                               (lg, None), (gg, []), (mg, ""), (res, []), (note, "")])

        with gr.Tab("CASIA · Identity") as t2:
            identity_tab("CASIA")
        with gr.Tab("SCface · Low-res identity") as t3:
            identity_tab("SCface")

        # -------- LFW (two-stage person -> photo) --------
        with gr.Tab("LFW · Verification") as t4:
            spec = JOBS["LFW"]
            all_persons = dsp.lfw_person_names()
            multi_persons = dsp.lfw_person_names(min_photos=2)
            gr.HTML(purpose_html(spec))
            gr.Markdown("For each side, pick a **person** (type to search), then one of **their "
                        "photos**. Verification is open-set — no LFW identity is a training class, "
                        "so any two of the 5,749 people's photos form a valid test.",
                        elem_classes="idmsg")

            # Someone with a single photo can only ever be paired against a *different* person, so
            # they cannot demonstrate a same-person match at all. Filtering them out leaves exactly
            # the 1,680 people the fine-tune's LFW head was built over.
            multi_only = gr.Checkbox(
                value=True, label=f"Only people with ≥2 photos  ·  {len(multi_persons):,} "
                                  f"of {len(all_persons):,}",
                info="Uncheck to include the one-photo people — they can only form "
                     "different-person pairs.")

            def person_choices(multi):
                names = multi_persons if multi else all_persons
                return (gr.update(choices=names, value=None), gr.update(choices=names, value=None),
                        gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                        None, None, "", "")

            def photo_choices(person):
                paths = dsp.lfw_person_photos(person)
                return gr.update(choices=[(os.path.basename(p), p) for p in paths], value=None), None, ""

            with gr.Row(equal_height=True):
                with gr.Column(elem_classes="card"):
                    pa = gr.Dropdown(choices=multi_persons, value=None, label="Face A · person", filterable=True,
                                     info=("Type to search a name." if multi_persons else "Run fetch_datasets.py to populate."))
                    pha = gr.Dropdown(choices=[], value=None, label="Face A · photo")
                    ia = gr.Image(type="pil", label="Face A", height=260)
                    sela = gr.Markdown("", elem_classes="selname")
                with gr.Column(elem_classes="card"):
                    pb = gr.Dropdown(choices=multi_persons, value=None, label="Face B · person", filterable=True,
                                     info=("Type to search a name." if multi_persons else "Run fetch_datasets.py to populate."))
                    phb = gr.Dropdown(choices=[], value=None, label="Face B · photo")
                    ib = gr.Image(type="pil", label="Face B", height=260)
                    selb = gr.Markdown("", elem_classes="selname")
            verdict = gr.Markdown()
            runv = run_button("Verify (both models)")
            res2 = gr.Dataframe(headers=RES_HEADERS, datatype="str", interactive=False,
                                row_count=(3, "fixed"), label="Resource cost", elem_classes="card")
            note2 = gr.Markdown("", elem_classes="note")

            multi_only.change(person_choices, inputs=[multi_only],
                              outputs=[pa, pb, pha, phb, ia, ib, sela, selb])
            pa.change(photo_choices, inputs=[pa], outputs=[pha, ia, sela])
            pb.change(photo_choices, inputs=[pb], outputs=[phb, ib, selb])
            pha.change(lambda p: (Image.open(p).convert("RGB"), f"Held-out · {os.path.basename(p)}") if p else (None, ""),
                       inputs=[pha], outputs=[ia, sela])
            phb.change(lambda p: (Image.open(p).convert("RGB"), f"Held-out · {os.path.basename(p)}") if p else (None, ""),
                       inputs=[phb], outputs=[ib, selb])
            runv.click(run_verify, inputs=[ia, ib, backend], outputs=[verdict, res2, note2])
            clearables.extend([(multi_only, True), (pa, None), (pha, None), (ia, None), (sela, ""),
                               (pb, None), (phb, None), (ib, None), (selb, ""),
                               (verdict, ""), (res2, []), (note2, "")])

        # -------- ImageNet (linear-probe classification; Base vs Ghost only) --------
        with gr.Tab("ImageNet · Linear probe") as t5:
            spec = JOBS["ImageNet"]
            gr.HTML(purpose_html(spec))
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes="card"):
                    dd, shown, more, _ = sample_picker("ImageNet", "Held-out ImageNet val sample")
                    img = gr.Image(type="filepath", label="Input image (or drop your own)", height=300)
                    sel = gr.Markdown("", elem_classes="selname")
                    with gr.Accordion("Data setup · why no Mamba arms", open=False):
                        gr.Markdown(spec["setup"], elem_classes="idmsg")
                with gr.Column(scale=1, elem_classes="card"):
                    lb = gr.Label(num_top_classes=TOP_K, label=ARM_NAME["baseline"], elem_classes="arm arm-base")
                    lg = gr.Label(num_top_classes=TOP_K, label=ARM_NAME["ghost"], elem_classes="arm arm-ghost")
            run = run_button()
            res = gr.Dataframe(headers=RES_HEADERS, datatype="str", interactive=False,
                               row_count=(3, "fixed"), label="Resource cost", elem_classes="card")
            note = gr.Markdown("", elem_classes="note")
            dd.change(lambda rel: load_sample("ImageNet", rel)[:2], inputs=[dd], outputs=[img, sel])
            img.upload(lambda p: (f"Dropped · `{os.path.basename(p)}`" if p else ""),
                       inputs=[img], outputs=[sel])
            run.click(run_classification, inputs=[img, backend], outputs=[lb, lg, res, note])
            clearables.extend([(dd, None), (img, None), (sel, ""), (lb, None), (lg, None),
                               (res, []), (note, "")])

        comps = [c for c, _ in clearables]
        resets = [v for _, v in clearables]
        for tab in (t1, t2, t3, t4, t5):
            tab.select(lambda: resets, inputs=None, outputs=comps)

        gr.Markdown("Reference metrics are the paper's multi-seed A5000 results; this demo runs one image "
                    "live, so latency is a batch-1 laptop measurement — comparable between the two models.",
                    elem_classes="note")
    return demo


if __name__ == "__main__":
    import gradio as gr

    _, launch_kw = _style_kwargs(gr)
    build_ui().launch(server_name="0.0.0.0", server_port=7860, **launch_kw)
