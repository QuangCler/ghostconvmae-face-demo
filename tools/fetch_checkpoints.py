#!/usr/bin/env python3
"""Download the fine-tuned face checkpoints into ./checkpoints/ (run once on your laptop).

Uses gdown against Google Drive — works on any normal internet connection. ~5.5 GB total.
Files already present (correct size) are skipped, so it is safe to re-run.

    pip install gdown
    python tools/fetch_checkpoints.py
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
DEST = os.path.join(HERE, "checkpoints")
os.makedirs(DEST, exist_ok=True)

# filename -> (Google Drive id, expected md5 or None). base/ghost fine-tuned face models.
FILES = {
    "celeba_baseline.pth": ("1p-6NGTMbEPEdOvTC6BDBYOUXMQ-sNPBx", "988e751f08b9c2118ffea052099c5d7a"),
    "celeba_ghost.pth":    ("1DGjuWOIA198UjTd-eRUh-5uzcrQwldZJ", "7d6b0f8460e2640a9d5ff45cdd0d4470"),
    "lfw_baseline.pth":    ("1blN2E1ZnOCqWA6Erijzw3rM3LdF38e2A", "00d59190459e857201794bf467aa4a4a"),
    "lfw_ghost.pth":       ("1xGMSrTdYmByEr2ik99kpbbzlWxDSbAbl", "cfde11f27fac0680a187e80611caf546"),
    "casia_ghost.pth":     ("1DeQ8YNvm4GQ1n3WIzcXwgEkJ4UAHDyTp", None),
    # SCface (cross-resolution identity, 130 classes) — the low-res stressor where Ghost's
    # compression costs most, mirroring the report's largest accuracy gap.
    #   base  = seed1 best_checkpoint (epoch 65). NOTE: seed0's uploaded best/last checkpoints
    #           were clobbered to an untrained epoch-0 state by a resume (head std ~0.0007 ->
    #           uniform "random" output); seed1 is the correct trained file.
    #   ghost = seed0 best_checkpoint (epoch 58).
    # md5s are pinned so a previously-downloaded broken file is re-fetched instead of skipped.
    "scface_baseline.pth": ("1EHLcKQu6LMwToFuNLZSoqO_vAcITYDCR", "c6d06602f3d69772e9f5df3d57c196ac"),
    "scface_ghost.pth":    ("1n4wKYnRO5HiIXW_S1rVvOvbwYxEJ_MOa", "6223c52f1437f7d4e638156f19807d33"),
    # casia_baseline.pth = freshly retrained demo checkpoint (40 ep, Top-1 80.6%); the report's
    # CASIA numbers are unchanged (they use the original 3-seed runs). Uploaded to the shared
    # finetune_result/baseline/casia/outputs/seed0_demo/ folder.
    "casia_baseline.pth":  ("15zYNKaqsLjApfmMGcbQrT37BFq3pLes9", None),
    # ImageNet-1K linear-probe heads: 90-epoch linear probe on top of the frozen 300-epoch
    # pretrained backbone (BatchNorm->Linear head; epoch 89 = 90th, epochs=90, blr=0.1, wd=0.0,
    # LARS, effective batch 512). Both are `best_checkpoint.pth` from the W&B project
    # `convmae-linprobe`: base = run convmae_base_lineprobe_resume90_from_ep39,
    # ghost = run allghost_ep300_linprobe_resume90_keep_lr. The two Mamba arms have probes too, but
    # they are CUDA/mamba_ssm-only and are not part of this laptop demo.
    # NB: the two Drive files were provided with base/ghost labels swapped; verified by each file's
    # own args.model (1eJal…=baseline, 1eRZ…=allghost) and pinned below so the right head loads.
    "imagenet_baseline.pth": ("1eJal666QqcArwYFuFHstsQaZMOnEx6AA", "4cb8538b70600a73ffda49e6ea63dbf5"),
    "imagenet_ghost.pth":    ("1eRZie7l-vIgVWe2unkQxaPwAg33QWTEz", "b1596ef269a06e68e15e832b56a98cb4"),
}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    try:
        import gdown
    except ImportError:
        print("Please `pip install gdown` first.", file=sys.stderr)
        sys.exit(1)

    ok, skipped, failed = [], [], []
    for name, (fid, want_md5) in FILES.items():
        dst = os.path.join(DEST, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            if want_md5 is None or md5(dst) == want_md5:
                print(f"[skip] {name} already present"); skipped.append(name); continue
        if fid == "FILL_ME_AFTER_UPLOAD":
            print(f"[pending] {name}: no Drive id yet (demo runs without it)."); continue
        print(f"[dl] {name} ...")
        try:
            gdown.download(id=fid, output=dst, quiet=False)
            if want_md5 and md5(dst) != want_md5:
                print(f"[warn] {name}: md5 mismatch (file may still work)")
            ok.append(name)
        except Exception as e:
            print(f"[fail] {name}: {e}"); failed.append(name)

    print(f"\nDone. downloaded={len(ok)} skipped={len(skipped)} failed={len(failed)}")
    if failed:
        print("Retry later for:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
