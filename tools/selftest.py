#!/usr/bin/env python3
"""End-to-end self-test: every task, every arm, every backend — through the demo's own buttons.

Two phases, both driven by the *same* entry points the UI calls, so what this prints is what a
click in the browser produces:

  A. Backend matrix — for each task and each backend (PyTorch / TensorRT FP16 / FP32) it presses
     "Run both models" ``--iters`` times on one fixed held-out sample. It reports the backend each
     arm actually ran on (TensorRT silently falls back, and the label is the only honest record of
     that), the resource row, whether the top-1 stayed identical across all iterations, and — for
     the TensorRT rows — the agreement gate against the PyTorch answer for the same image.

  B. Accuracy sweep — a held-out sample per task, scored with that task's own metric and compared
     against the number the README/report claims. This is the part that catches a *silently* wrong
     pipeline: a mislabelled dataset or a swapped checkpoint still runs at full speed and still
     prints confident answers, and only the metric shows it.

Each task is torn down before the next (PyTorch cache, TensorRT engine cache, CUDA allocator),
because a 4 GB laptop card cannot hold ten backbones plus five pairs of engines at once.

    python tools/selftest.py                       # full run
    python tools/selftest.py --iters 50 --samples 400
    python tools/selftest.py --jobs CASIA ImageNet --skip-accuracy
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import app                                    # noqa: E402
from demo import dataset_paths as dsp         # noqa: E402
from demo import trt_backend                  # noqa: E402

ARMS = ("baseline", "ghost")
BACKENDS = list(app.BACKENDS)
JOBS = list(app.JOBS)

# What the README / report claims, per (job, arm). Checked in phase B; `tol` is how far the live
# laptop measurement may drift before the run is called a mismatch rather than noise.
EXPECTED = {
    "CelebA":   {"metric": "mAP",   "baseline": 0.789,  "ghost": 0.778,  "tol": 0.05},
    "CASIA":    {"metric": "top-1", "baseline": 0.908,  "ghost": 0.905,  "tol": 0.05},
    # SCface is scored on the VISIBLE cameras: that is the report's setting, and the picker also
    # offers the two infrared cameras (~13% / ~7%) plus the IR mugshot, which drag the pooled
    # number down to ~36% / ~26% without saying anything about the model.
    "SCface":   {"metric": "top-1 visible", "baseline": 0.4513, "ghost": 0.3136, "tol": 0.05},
    "LFW":      {"metric": "AUC",   "baseline": 0.9921, "ghost": 0.9833, "tol": 0.02},
    "ImageNet": {"metric": "top-1", "baseline": 0.6406, "ghost": 0.5860, "tol": 0.04},
}


# ---------------------------------------------------------------- metrics (no sklearn dependency)
def _avg_ranks(x):
    """Ranks 1..n with ties averaged — what a tie-correct AUC needs."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    n1, n0 = int((labels == 1).sum()), int((labels == 0).sum())
    if not n1 or not n0:
        return float("nan")
    r = _avg_ranks(scores)
    return (r[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def average_precision(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order]
    prec = np.cumsum(lab) / np.arange(1, len(lab) + 1)
    return float((prec * lab).sum() / lab.sum())


# ---------------------------------------------------------------- helpers
def teardown():
    """Drop every cached model and engine. Ten backbones will not fit in 4 GB."""
    app._CACHE.clear()
    app._WARMED.clear()
    app._LATENCY.clear()
    trt_backend._ENGINES.clear()
    if app.CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def pick_samples(job, n, seed=0):
    """A deterministic held-out subset — same images every run, so numbers are comparable."""
    items = sorted(dsp.test_samples(job).items())
    rnd = random.Random(seed)
    return rnd.sample(items, min(n, len(items)))


def lfw_pairs(n_each, seed=0):
    """n_each same-person pairs + n_each different-person pairs, from the ≥2-photo people."""
    rnd = random.Random(seed)
    people = dsp.lfw_person_names(min_photos=2)
    out = []
    for _ in range(n_each):
        p = rnd.choice(people)
        a, b = rnd.sample(dsp.lfw_person_photos(p), 2)
        out.append((a, b, 1))
    for _ in range(n_each):
        p, q = rnd.sample(people, 2)
        out.append((rnd.choice(dsp.lfw_person_photos(p)),
                    rnd.choice(dsp.lfw_person_photos(q)), 0))
    return out


def table_rows(table):
    """{arm: {backend, latency_ms, vram_MB, params_M}} out of the UI's resource dataframe."""
    out = {}
    for arm, row in zip(ARMS, table):
        out[arm] = {"backend": row[1], "params_M": row[2], "weights_MB": row[3],
                    "latency_ms": row[4], "vram_MB": row[5]}
    return out


def top1(label_dict):
    """(class name, probability) from a gr.Label payload, or (None, None) when empty."""
    if not label_dict:
        return None, None
    k = next(iter(label_dict))
    return k, float(label_dict[k])


# ---------------------------------------------------------------- phase A: backend matrix
def press_run(job, backend, sample):
    """One press of that tab's button. Returns ({arm: (top1, prob)}, resource table, extra)."""
    path = sample[1] if isinstance(sample, tuple) else sample
    if job == "CelebA":
        lb, lg, gt, table, _ = app.run_attribute(path, backend, sample[0])
        return {"baseline": top1(lb), "ghost": top1(lg)}, table, {"truth": gt[:60]}
    if job == "ImageNet":
        lb, lg, gt, table, _ = app.run_classification(path, backend, sample[0])
        return {"baseline": top1(lb), "ghost": top1(lg)}, table, {"truth": gt[:60]}
    if job == "LFW":
        a, b = sample
        body, table, _ = app.run_verify(Image.open(a).convert("RGB"),
                                        Image.open(b).convert("RGB"), backend)
        # The verdict card is the tab's whole answer, so parse it back rather than re-running the
        # model: "- **ConvMAE-Base** — cosine `+0.412` vs threshold `0.235` → **SAME person**".
        preds = {}
        for arm in ARMS:
            m = re.search(re.escape(app.ARM_NAME[arm]) + r".*?cosine `([-+0-9.]+)`.*?\*\*(.+?)\*\*",
                          body)
            preds[arm] = (m.group(2), float(m.group(1))) if m else (None, None)
        return preds, table, {"verdict": body.splitlines()[0][:80]}
    lb, gb, mb, lg, gg, mg, table, _ = app.run_identity(job, path, backend)
    return ({"baseline": top1(lb), "ghost": top1(lg)}, table,
            {"face_baseline": bool(gb), "face_ghost": bool(gg)})


def phase_a(job, iters, sample):
    print(f"\n=== {job} — backend matrix ({iters} presses of “Run both models” per backend) ===")
    results, ref = {}, None
    for backend in BACKENDS:
        t0 = time.perf_counter()
        preds, table, extra, err = None, None, None, None
        seen = {a: set() for a in ARMS}
        try:
            for _ in range(iters):
                preds, table, extra = press_run(job, backend, sample)
                for a in ARMS:
                    seen[a].add(preds[a][0])
        except Exception as e:                       # a broken backend must not kill the sweep
            err = f"{type(e).__name__}: {e}"
        wall = time.perf_counter() - t0
        if err:
            print(f"  {backend:18s} ERROR  {err}")
            results[backend] = {"error": err}
            continue
        rows = table_rows(table)
        results[backend] = {"rows": rows, "preds": {a: preds[a] for a in ARMS},
                            "stable": {a: len(seen[a]) == 1 for a in ARMS},
                            "wall_s": round(wall, 1), **extra}
        if backend == "PyTorch":
            ref = preds
        for a in ARMS:
            r = rows[a]
            name, prob = preds[a]
            shown = f"{name} {prob * 100:.1f}%" if name and prob is not None else "—"
            gate = ""
            if ref and backend != "PyTorch" and name is not None:
                agree = "top-1 ✓" if name == ref[a][0] else f"top-1 ✗ (PyTorch said {ref[a][0]})"
                dp = abs(prob - ref[a][1]) if prob is not None and ref[a][1] is not None else float("nan")
                gate = f" | vs PyTorch: {agree}, Δp {dp:.4f}"
            stab = "" if results[backend]["stable"][a] else "  ⚠ TOP-1 DRIFTED ACROSS ITERATIONS"
            print(f"  {backend:18s} {a:8s} ran={r['backend']:22s} "
                  f"{r['latency_ms']:>6s} ms  {r['vram_MB']:>5s} MB  {shown}{gate}{stab}")
        for k, v in extra.items():
            if k.startswith("face_") and not v:
                print(f"  {backend:18s} {'':8s} ⚠ no predicted-person image resolved ({k})")
    return results


# ---------------------------------------------------------------- phase B: accuracy sweep
@torch.no_grad()
def accuracy(job, n):
    """Score `n` held-out samples with the task's own metric, both arms, PyTorch backend."""
    if job == "LFW":
        pairs = lfw_pairs(n // 2)
        cos = {a: [] for a in ARMS}
        labels = [p[2] for p in pairs]
        for a, b, _ in pairs:
            xa = app.PREPROC(Image.open(a).convert("RGB")).unsqueeze(0).to(app.DEVICE)
            xb = app.PREPROC(Image.open(b).convert("RGB")).unsqueeze(0).to(app.DEVICE)
            x = torch.cat([xa, xb])
            for arm in ARMS:
                out, *_ = app.infer_one("LFW", arm, x, "PyTorch", feats=True)
                e = torch.nn.functional.normalize(out.float(), dim=-1)
                cos[arm].append(float((e[0] * e[1]).sum().cpu()))
        res = {}
        for arm in ARMS:
            s = np.asarray(cos[arm])
            acc = float(((s > app.LFW_THRESHOLD).astype(int) == np.asarray(labels)).mean())
            res[arm] = {"AUC": roc_auc(s, labels), "acc@thr": acc, "n": len(pairs)}
        return res

    samples = pick_samples(job, n)
    if job == "CelebA":
        truth_map = app.celeba_truth()
        probs = {a: [] for a in ARMS}
        truths = []
        for rel, path in samples:
            t = truth_map.get(os.path.basename(rel))
            if t is None:
                continue
            truths.append([1 if attr in t else 0 for attr in app.CELEBA_ATTRS])
            x = app.PREPROC(Image.open(path).convert("RGB")).unsqueeze(0).to(app.DEVICE)
            for arm in ARMS:
                out, *_ = app.infer_one("CelebA", arm, x, "PyTorch")
                probs[arm].append(torch.sigmoid(out)[0].float().cpu().numpy())
        Y = np.asarray(truths)
        res = {}
        for arm in ARMS:
            P = np.asarray(probs[arm])
            aps = [average_precision(P[:, k], Y[:, k]) for k in range(Y.shape[1])]
            aps = [v for v in aps if not np.isnan(v)]
            res[arm] = {"mAP": float(np.mean(aps)),
                        "attr_acc": float(((P > app.ATTR_THRESHOLD).astype(int) == Y).mean()),
                        "n": len(Y)}
        return res

    # SCface's picker mixes visible and infrared cameras, and the report's number is the visible
    # ones only, so score both: "top-1" over everything offered, "top-1 visible" over cam1-cam5.
    hits = {(a, g): [0, 0, 0] for a in ARMS for g in ("all", "visible")}
    for rel, path in samples:
        truth = job_truth(job, rel)
        if truth is None:
            continue
        groups = ["all"] + (["visible"] if job == "SCface" and "IR" not in rel else [])
        x = app.PREPROC(Image.open(path).convert("RGB")).unsqueeze(0).to(app.DEVICE)
        for arm in ARMS:
            out, *_ = app.infer_one(job, arm, x, "PyTorch")
            top = out.float().topk(5, dim=-1).indices[0].tolist()
            for g in groups:
                h = hits[(arm, g)]
                h[0] += int(top[0] == truth)
                h[1] += int(truth in top)
                h[2] += 1

    def rate(arm, g, i):
        h = hits[(arm, g)]
        return h[i] / h[2] if h[2] else float("nan")

    res = {}
    for a in ARMS:
        res[a] = {"top-1": rate(a, "all", 0), "top-5": rate(a, "all", 1), "n": hits[(a, "all")][2]}
        if job == "SCface":
            res[a]["top-1 visible"] = rate(a, "visible", 0)
            res[a]["n_visible"] = hits[(a, "visible")][2]
    return res


def job_truth(job, rel):
    """The true class INDEX for a held-out sample, or None when the label cannot be resolved."""
    if job == "ImageNet":
        t = dsp.imagenet_truth(rel)
        return None if t is None else t[0]
    if job == "CASIA":                       # rel is '<label folder>/<file>'
        names = dsp.class_names("CASIA")
        folder = rel.split("/")[0]
        return names.index(folder) if folder in names else None
    if job == "SCface":                      # rel is '001 · cam1 · d1'; subjects are 1-based
        try:
            return int(rel.split(" ")[0]) - 1
        except ValueError:
            return None
    return None


def phase_b(job, n):
    print(f"\n=== {job} — accuracy on {n} held-out samples (PyTorch) ===")
    res = accuracy(job, n)
    exp = EXPECTED[job]
    key = exp["metric"] if exp["metric"] in next(iter(res.values())) else \
        next(iter(next(iter(res.values()))))
    for arm in ARMS:
        got = res[arm].get(key, float("nan"))
        want = exp[arm]
        delta = got - want
        flag = "ok" if abs(delta) <= exp["tol"] else "MISMATCH"
        extras = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in res[arm].items() if k != key)
        print(f"  {arm:8s} {key}={got:.4f}  expected {want:.4f}  Δ{delta:+.4f}  [{flag}]   {extras}")
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", default=JOBS, choices=JOBS)
    ap.add_argument("--iters", type=int, default=50, help="button presses per backend")
    ap.add_argument("--samples", type=int, default=400, help="held-out images per accuracy check")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--skip-matrix", action="store_true")
    ap.add_argument("--out", default=None, help="write the raw results as JSON here")
    args = ap.parse_args()

    print(f"device={app.DEVICE}  cuda={app.CUDA}  tensorrt={trt_backend.available()}")
    if app.CUDA:
        print(f"gpu={torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    report = {}
    for job in args.jobs:
        teardown()
        entry = {}
        if not args.skip_matrix:
            sample = (lfw_pairs(1)[0][:2] if job == "LFW" else pick_samples(job, 1)[0])
            entry["matrix"] = phase_a(job, args.iters, sample)
            teardown()
        if not args.skip_accuracy:
            entry["accuracy"] = phase_b(job, args.samples)
        report[job] = entry
    teardown()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nraw results -> {args.out}")


if __name__ == "__main__":
    main()
