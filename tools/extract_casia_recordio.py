#!/usr/bin/env python3
"""Unpack CASIA-WebFace from the InsightFace RecordIO into the fine-tune's ImageFolder layout.

    datasets/casia/prepared/train/<label:05d>/<key>.jpg
    datasets/casia/prepared/val/<label:05d>/<key>.jpg     (VAL_PER_CLASS held out per identity)

The identity of a record comes from its **IRHeader label**, which is the only authoritative source
and is what the fine-tune's own prepare_casia.py used (see assets/casia_class_maps/). An earlier
version of this script instead paired record N with line N of `train.lst` — that is wrong, and
wrong in a way that silently corrupts every label:

    train.idx holds 501,196 records; train.lst holds 494,149 lines, in a different order.
    They agree at the start and drift apart: at key 5,000 the record header says label 28 while
    line 5,000 says 27; by key 341,263 it is 6,332 against 6,285.

So the old output put each identity's images into a neighbouring identity's folder, with the error
growing through the file. That is what made the demo's predicted-person face wrong — the folders it
read were mislabelled, not the model.

The Kaggle zip unpacks with the payload nested one level down, and a half-finished unzip leaves a
truncated `train.rec` in the outer directory next to the complete one. Reading the short file just
drops every record past the cut, so --raw picks whichever candidate directory holds a `train.rec`
long enough to cover its own index. Rerunning skips images already written, so extraction resumes
safely after an interruption.
"""
from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

_MAGIC = 0xCED7230A
_HEADER = struct.Struct("<IfQQ")        # flag, label, id, id2
_PREFIX = struct.Struct("<II")          # magic, lrecord
VAL_PER_CLASS = 2                       # matches prepare_casia.py's --val_per_class default


def read_index(path: Path) -> list[tuple[int, int]]:
    """train.idx lines are '<key>\\t<offset>'. Returns [(key, offset)] sorted by key."""
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                key, offset = line.split("\t")
                out.append((int(key), int(offset)))
            except ValueError as exc:
                raise ValueError(f"Malformed index at line {line_no}: {line!r}") from exc
    out.sort()
    return out


def read_record(fh, offset: int) -> tuple[int, bytes]:
    """Return (label, image bytes) for the record at `offset`."""
    fh.seek(offset)
    prefix = fh.read(_PREFIX.size)
    if len(prefix) != _PREFIX.size:
        raise EOFError("truncated record prefix")
    magic, lrecord = _PREFIX.unpack(prefix)
    if magic != _MAGIC:
        raise ValueError(f"bad record magic {magic:#x} at offset {offset}")
    length = lrecord & ((1 << 29) - 1)      # top 3 bits are a continuation flag
    data = fh.read(length)
    if len(data) != length:
        raise EOFError("truncated record payload")
    flag, label, _id, _id2 = _HEADER.unpack(data[:_HEADER.size])
    payload = data[_HEADER.size + flag * 4:] if flag > 0 else data[_HEADER.size:]
    return int(label), payload


def pick_raw(base: Path) -> Path:
    """Prefer a copy whose train.rec actually covers its index over a truncated sibling."""
    best, best_score = None, -1
    for cand in (base, base / base.name, *sorted(p for p in base.glob("*") if p.is_dir())):
        rec, idx = cand / "train.rec", cand / "train.idx"
        if not (rec.is_file() and idx.is_file()):
            continue
        size = rec.stat().st_size
        needed = max(o for _, o in read_index(idx))
        score = (1 if size > needed else 0, size)
        if score > (best_score if isinstance(best_score, tuple) else (-1, -1)):
            best, best_score = cand, score
    return best or base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("datasets/casia/casia-webface"))
    ap.add_argument("--out", type=Path, default=Path("datasets/casia/prepared"))
    ap.add_argument("--val-per-class", type=int, default=VAL_PER_CLASS)
    ap.add_argument("--limit", type=int, help="stop after this many images (for a quick check)")
    args = ap.parse_args()

    raw, out = pick_raw(args.raw.resolve()), args.out.resolve()
    rec_path, idx_path = raw / "train.rec", raw / "train.idx"
    for p in (rec_path, idx_path):
        if not p.is_file():
            raise SystemExit(f"Missing input: {p}")
    print(f"reading {rec_path} ({rec_path.stat().st_size:,} bytes)")

    size = rec_path.stat().st_size
    index = read_index(idx_path)
    usable = [(k, o) for k, o in index if k > 0 and 0 <= o < size]
    if len(usable) < len(index) - 1:
        print(f"[warn] train.rec is {size:,} bytes but {len(index) - 1 - len(usable):,} of "
              f"{len(index) - 1:,} records point past its end — the file is incomplete. "
              f"Re-download casia-webface to extract the rest.")

    per_class: dict[int, int] = {}
    written = skipped = failed = 0
    with rec_path.open("rb") as fh:
        for key, offset in usable:
            if args.limit and written + skipped >= args.limit:
                break
            try:
                label, image = read_record(fh, offset)
            except (EOFError, ValueError):
                failed += 1
                continue
            if image[:2] != b"\xff\xd8":        # meta/index records are not JPEGs
                failed += 1
                continue
            seen = per_class.get(label, 0)
            split = "val" if seen < args.val_per_class else "train"
            target = out / split / f"{label:05d}" / f"{key}.jpg"
            per_class[label] = seen + 1
            if target.is_file() and target.stat().st_size > 0:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(image)
            written += 1
            if written % 20000 == 0:
                print(f"  {written:,} written, {len(per_class):,} identities", flush=True)

    print(f"\nDone. written={written:,} skipped={skipped:,} unusable={failed:,} "
          f"identities={len(per_class):,} -> {out}")


if __name__ == "__main__":
    main()
