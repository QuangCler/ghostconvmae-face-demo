#!/usr/bin/env python3
"""Split the flat CelebA image folder into train/ and test/ by the official partition file.

    datasets/celeba/train/<image_id>.jpg     partition 0            (162,770 images, fitted on)
    datasets/celeba/test/<image_id>.jpg      partitions 1 and 2     ( 39,829 images, held out)

`list_eval_partition.csv` is the authority; the split is not inferred from filenames. Files are
moved, not copied, so the ~1.4 GB is not duplicated, and a file already in place is left alone —
rerun freely after an interruption.

    python tools/split_celeba.py              # move into place
    python tools/split_celeba.py --copy       # keep the flat folder intact as well
    python tools/split_celeba.py --check      # report what would move, change nothing
"""
import argparse
import csv
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent            # repo root
CELEBA = HERE / "datasets" / "celeba"
PARTITION = {"0": "train", "1": "test", "2": "test"}     # 1 (val) and 2 (test) are both held out


def flat_root() -> Path | None:
    for c in (CELEBA / "img_align_celeba" / "img_align_celeba", CELEBA / "img_align_celeba"):
        if c.is_dir() and any(f.suffix.lower() == ".jpg" for f in list(c.iterdir())[:20]):
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    args = ap.parse_args()

    part_csv = CELEBA / "list_eval_partition.csv"
    if not part_csv.is_file():
        raise SystemExit(f"Missing {part_csv} — run `python tools/fetch_datasets.py --only celeba`.")
    src = flat_root()

    with part_csv.open(newline="") as f:
        r = csv.reader(f)
        next(r, None)
        rows = [(row[0].strip(), PARTITION.get(row[1].strip())) for row in r if len(row) >= 2]

    moved = present = missing = 0
    for name, split in rows:
        if not split:
            continue
        dst = CELEBA / split / name
        if dst.is_file():
            present += 1
            continue
        if src is None or not (src / name).is_file():
            missing += 1
            continue
        moved += 1
        if args.check:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copy2 if args.copy else shutil.move)(str(src / name), str(dst))
        if moved % 20000 == 0:
            print(f"  {moved:,} placed", flush=True)

    verb = "would move" if args.check else ("copied" if args.copy else "moved")
    print(f"\n{verb}={moved:,} already_in_place={present:,} not_found={missing:,}")
    for split in ("train", "test"):
        d = CELEBA / split
        n = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
        print(f"  {split}: {n:,} images -> {d}")
    if missing and not args.check:
        print("\n[warn] some images listed in the partition file were not on disk — "
              "re-run `python tools/fetch_datasets.py --only celeba` if the count looks wrong.")


if __name__ == "__main__":
    main()
