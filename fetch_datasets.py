#!/usr/bin/env python3
"""Download the original face datasets used by the project (into ./datasets/).

Uses the Kaggle API for the openly-downloadable sets. These are large — grab only what you
need with --only.

    LFW      jessicali9530/lfw-dataset       (~0.2 GB)   public benchmark
    CelebA   jessicali9530/celeba-dataset    (~1.4 GB)   research use
    CASIA    debarghamitraroy/casia-webface  (~4 GB)     research use
    SCface   — access-controlled — not downloadable here (request a licence; see below)
    imagenet — the ImageNet-1K val set for the linear-probe tab. It is the Kaggle *competition*
               `imagenet-object-localization-challenge` (LARGE: ~155 GB archive; the val split we
               use is ~6.4 GB). Downloaded only on explicit `--only imagenet`. If you already have
               val on disk, set IMAGENET_VAL_DIR instead and skip the download entirely.

Setup once:
    pip install kaggle
    # put your token at ~/.kaggle/kaggle.json  (Kaggle -> Account -> Create New API Token)
    #   or export KAGGLE_USERNAME=... KAGGLE_KEY=...

Usage:
    python fetch_datasets.py                 # LFW + CelebA + CASIA (faces)
    python fetch_datasets.py --only lfw      # just one
    python fetch_datasets.py --only imagenet # ImageNet-1K val for the linear-probe tab (large)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "datasets")

KAGGLE = {
    "lfw":    "jessicali9530/lfw-dataset",
    "celeba": "jessicali9530/celeba-dataset",
    "casia":  "debarghamitraroy/casia-webface",
}


def fetch_imagenet_val(dest):
    """ImageNet-1K val for the linear-probe tab. NOT a Kaggle dataset — the competition
    `imagenet-object-localization-challenge` (~155 GB archive; val ≈ 6.4 GB). Prefer reusing an
    existing copy: set IMAGENET_VAL_DIR and this is skipped entirely."""
    out = os.path.join(dest, "imagenet")
    val = os.path.join(out, "ILSVRC", "Data", "CLS-LOC", "val")
    if os.path.isdir(val) and os.listdir(val):
        print(f"[skip] imagenet: {val}/ already populated"); return
    if os.environ.get("IMAGENET_VAL_DIR"):
        print("[skip] imagenet: IMAGENET_VAL_DIR is set — the app reads val from there, no download needed.")
        return
    print("[imagenet] This downloads the Kaggle competition "
          "`imagenet-object-localization-challenge` — LARGE (~155 GB archive; the val split we need "
          "is ~6.4 GB), and you must have accepted its rules on kaggle.com first.\n"
          "           To avoid it: Ctrl-C now and either `export IMAGENET_VAL_DIR=/path/to/val` or "
          f"drop an existing val/ folder at {val}.")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print("Please `pip install kaggle` and set up your token.", file=sys.stderr); sys.exit(1)
    api = KaggleApi(); api.authenticate()
    os.makedirs(out, exist_ok=True)
    api.competition_download_files("imagenet-object-localization-challenge", path=out, quiet=False)
    import glob
    import zipfile
    for z in glob.glob(os.path.join(out, "*.zip")):
        print(f"[imagenet] unzipping {os.path.basename(z)} (this takes a while) ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    if os.path.isdir(val):
        print(f"[ok] imagenet val -> {val}/")
    else:
        print(f"[imagenet] archive extracted under {out}/ — expected val at {val}; check the layout.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(KAGGLE) + ["imagenet"],
                    help="subset to download (imagenet is large and download-only on request)")
    args = ap.parse_args()
    names = args.only or list(KAGGLE)          # faces by default; imagenet only via --only imagenet
    if "imagenet" in names:
        fetch_imagenet_val(DEST)
        names = [n for n in names if n != "imagenet"]
        if not names:
            return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print("Please `pip install kaggle` and set up your Kaggle token "
              "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).", file=sys.stderr)
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"Kaggle auth failed: {e}\nCreate a token at Kaggle -> Account -> Create New API Token.",
              file=sys.stderr)
        sys.exit(1)

    for name in names:
        out = os.path.join(DEST, name)
        os.makedirs(out, exist_ok=True)
        if os.listdir(out):
            print(f"[skip] {name}: {out}/ already populated"); continue
        print(f"[dl] {name}  <-  {KAGGLE[name]}  (this can be large)")
        api.dataset_download_files(KAGGLE[name], path=out, unzip=True, quiet=False)
        print(f"[ok] {name} -> {out}/")

    print("\nSCface is access-controlled and cannot be auto-downloaded. Request a research licence at "
          "https://www.scface.org/ and place the images under datasets/scface/. The report uses SCface "
          "strictly under its licence and redistributes no personal data.")
    print(f"\nDatasets under {DEST}/  — drop any face image into the app to test.")


if __name__ == "__main__":
    main()
