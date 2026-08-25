#!/usr/bin/env python3
"""Calibrate the LFW same/different threshold on the official pairs protocol.

The demo shipped a hardcoded `cosine > 0.5`, which is far too strict for these embeddings: it
recognises essentially no false matches (TNR 100%) but misses over half the true ones (TPR ~50%),
so a pair of genuine photos of the same person usually reads "DIFFERENT". The separation itself is
fine — ROC-AUC is ~0.99 — the cut was simply in the wrong place.

This measures where it belongs, per arm, on the 6,000 official `pairs.csv` pairs, and writes
`lfw_thresholds.json` for the app to load. Each image is embedded once and reused across every
pair it appears in, so the whole sweep is a few thousand forwards rather than 12,000.

    python tools/calibrate_lfw.py                 # both arms -> lfw_thresholds.json
    python tools/calibrate_lfw.py --limit 1000    # quick check on a subset of pairs
"""
import argparse
import csv
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from demo import dataset_paths as dsp  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
OUT = os.path.join(HERE, "lfw_thresholds.json")


def official_pairs():
    """[(path_a, path_b, is_same)] from pairs.csv: 'name,n1,n2' matched, 'n1,i,n2,j' mismatched."""
    path = os.path.join(dsp.DATASETS, "lfw", "pairs.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"Missing {path} — run `python tools/fetch_datasets.py --only lfw`.")
    root = dsp.lfw_root()

    def img(person, num):
        p = os.path.join(root, person, f"{person}_{int(num):04d}.jpg")
        return p if os.path.isfile(p) else None

    out = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            row = [c.strip() for c in row]
            if len(row) < 3 or not row[0]:
                continue
            if len(row) >= 4 and row[3]:
                a, b, same = img(row[0], row[1]), img(row[2], row[3]), False
            else:
                a, b, same = img(row[0], row[1]), img(row[0], row[2]), True
            if a and b:
                out.append((a, b, same))
    return out


@torch.no_grad()
def embed_all(model, paths, device, preproc, batch):
    """{path: L2-normalised embedding} — one forward per distinct image, not per pair."""
    paths = sorted(set(paths))
    table = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        x = torch.stack([preproc(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        e = torch.nn.functional.normalize(model.forward_features(x).float(), dim=-1).cpu()
        table.update(zip(chunk, e))
        if (i // batch) % 40 == 0:
            print(f"  {min(i + batch, len(paths))}/{len(paths)} images", flush=True)
    return table


def sweep(scores, labels):
    """Best accuracy over a fine threshold grid, plus the ROC-AUC of the same scores."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    best = max(((sum(s > t for s in pos) + sum(s <= t for s in neg)) / len(scores), t)
               for t in (i / 500 for i in range(-500, 500)))
    auc = sum(1 for a in pos for b in neg if a > b) / (len(pos) * len(neg))
    return best[1], best[0], auc, pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, help="use only the first N pairs")
    args = ap.parse_args()

    import app
    pairs = official_pairs()[:args.limit]
    n_pos = sum(1 for *_, s in pairs if s)
    print(f"{len(pairs)} official pairs ({n_pos} same, {len(pairs) - n_pos} different)")

    result = {}
    for arm in ("baseline", "ghost"):
        model, _ = app.get_model("LFW", arm)
        table = embed_all(model, [p for a, b, _ in pairs for p in (a, b)],
                          app.DEVICE, app.PREPROC, args.batch)
        scores = [float((table[a] * table[b]).sum()) for a, b, _ in pairs]
        labels = [s for *_, s in pairs]
        thr, acc, auc, pos, neg = sweep(scores, labels)
        at_half = (sum(s > 0.5 for s in pos) + sum(s <= 0.5 for s in neg)) / len(scores)
        result[arm] = {"threshold": round(thr, 3), "accuracy": round(acc, 4),
                       "auc": round(auc, 4), "pairs": len(pairs)}
        print(f"\n{app.ARM_NAME[arm]}: best acc {acc * 100:.1f}% at cosine > {thr:.3f} "
              f"| AUC {auc:.4f} | the old fixed 0.5 gave {at_half * 100:.1f}%")

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
