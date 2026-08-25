"""Explicit dataset layout for the demo — where the held-out images and the train faces live.

This replaces an earlier version that sniffed directory trees at runtime, scoring candidates by how
many subfolders held images. That guess mapped SCface onto its seven `cam_*` folders and CASIA onto
however many identities happened to be extracted — neither matching the checkpoint's head — so the
demo silently fell back to bare index numbers and never showed a predicted-person face. Every path
is declared here instead.

Three functions carry the whole contract:

    test_samples(job)     -> {label: path}   held-out images ONLY — nothing the model trained on
    class_label(job, idx) -> name | None     what a predicted class index is called
    class_face(job, idx)  -> path | None     a TRAIN image of that class, for "predicted person"

What counts as held-out, mirroring the fine-tune's data preparation:

    CelebA   list_eval_partition.csv partition 1 (val) + 2 (test); partition 0 was trained on.
    CASIA    the first VAL_PER_CLASS images of each identity (prepare_casia.py:
             `split = "val" if c < val_per_class`); images from index 2 on were trained on.
    SCface   the surveillance crops. The model only ever saw the high-res mugshots, so every
             surveillance image is out-of-domain — this is the strictest held-out set of the four.
    LFW      the 6000 official pairs. Verification is open-set, so the pair protocol *is* the test.

Class-index order is the ImageFolder convention used by the fine-tune: `sorted(folder_names)`.
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root, one level up
DATASETS = os.path.join(ROOT, "datasets")
ASSETS = os.path.join(ROOT, "assets")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CASIA_VAL_PER_CLASS = 2          # prepare_casia.py held out the first 2 images of each identity
SCFACE_SUBJECTS = 130
PAGE = 300                       # dropdown shows this many at a time and pages in more on demand

_cache = {}


def _cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def _first_dir(*candidates):
    """First existing directory among the candidates (datasets unzip with varying nesting)."""
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


# ------------------------------------------------------------------ CelebA
def celeba_split(split):
    """`datasets/celeba/<split>/` — written by split_celeba.py (train = partition 0)."""
    return _first_dir(os.path.join(DATASETS, "celeba", split))


def celeba_root():
    """Fallback flat folder, for a dataset that has not been split yet."""
    return _first_dir(os.path.join(DATASETS, "celeba", "img_align_celeba", "img_align_celeba"),
                      os.path.join(DATASETS, "celeba", "img_align_celeba"))


def celeba_attr_csv():
    p = os.path.join(DATASETS, "celeba", "list_attr_celeba.csv")
    return p if os.path.isfile(p) else None


def celeba_partition(basename):
    """'train' / 'val' / 'test' for a CelebA filename, or None if it is not a CelebA image.

    Worth showing for a dropped image: partition 0 is what the model was fitted on, so a high
    score there says much less than the same score on 1 or 2.
    """
    def build():
        path = os.path.join(DATASETS, "celeba", "list_eval_partition.csv")
        if not os.path.isfile(path):
            return {}
        name = {"0": "train", "1": "val", "2": "test"}
        out = {}
        with open(path, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 2:
                    out[row[0].strip()] = name.get(row[1].strip())
        return out
    return _cached("celeba_partition", build).get(os.path.basename(basename))


def _celeba_test():
    """{'val/162771.jpg': path} — the held-out images, partitions 1 and 2.

    Prefers `datasets/celeba/test/` once split_celeba.py has run; otherwise it filters the flat
    folder through the partition file, so the tab behaves the same either way.
    """
    test_dir = celeba_split("test")
    part = os.path.join(DATASETS, "celeba", "list_eval_partition.csv")
    if not os.path.isfile(part):
        return {}
    split_name = {"1": "val", "2": "test"}
    rows = []
    with open(part, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2 and row[1].strip() in split_name:
                rows.append((split_name[row[1].strip()], row[0].strip()))
    root = test_dir or celeba_root()
    if not root:
        return {}
    return {f"{sp}/{name}": os.path.join(root, name) for sp, name in rows}


# ------------------------------------------------------------------ CASIA
def casia_split(split):
    """`datasets/casia/prepared/<split>/` — written by extract_casia_recordio.py."""
    return _first_dir(os.path.join(DATASETS, "casia", "prepared", split))


def _casia_classes():
    """The fine-tune's exact class order: classes[i] is the folder for class index i.

    It is `ImageFolder(train_dir).classes` from the run itself, so no reconstruction is involved.
    Folder names are the RecordIO integer labels zero-padded to 5 digits, and they are NOT
    contiguous: label 09282 has only 2 images, both of which went to val, so it never reached
    train/ — class index i maps to label i below 9282 and i+1 from 9282 on.

    `datasets/` is not in version control, so the copy under assets/ is the one a fresh clone gets;
    reading it directly keeps the CASIA tab working without the dataset having to be present.
    """
    for path in (os.path.join(DATASETS, "casia", "class_order.json"),
                 os.path.join(ASSETS, "casia_class_maps", "casia_classes_baseline.json")):
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                continue
    root = casia_split("train")
    return sorted(os.listdir(root)) if root else []


def _identity_images(root, name):
    d = os.path.join(root, name)
    try:
        return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))
    except OSError:
        return []


def casia_extracted():
    """Identity folders actually on disk — extraction is resumable, so this can lag the class list."""
    def build():
        root = casia_split("train")
        if not root:
            return []
        present = set(os.listdir(root))
        return [n for n in class_names("CASIA") if n in present]
    return _cached("casia_extracted", build)


def _casia_test():
    """{'00042/1234.jpg': path} — the held-out val split, straight from the fine-tune's own layout."""
    root = casia_split("val")
    if not root:
        return {}
    out = {}
    for nm in sorted(os.listdir(root)):
        for fn in _identity_images(root, nm):
            out[f"{nm}/{fn}"] = os.path.join(root, nm, fn)
    return out


def _casia_train_face(name):
    root = casia_split("train")
    imgs = _identity_images(root, name) if root else []
    return os.path.join(root, name, imgs[0]) if imgs else None


# ------------------------------------------------------------------ SCface
def scface_mugshots():
    return _first_dir(os.path.join(DATASETS, "scface", "mugshot_frontal_cropped_all"),
                      os.path.join(DATASETS, "scface", "mugshot_frontal_original_all"))


def scface_surveillance():
    return _first_dir(os.path.join(DATASETS, "scface", "surveillance_cameras_all"))


SCFACE_IR_CAMS = ("cam6", "cam7")     # SCface's two night/IR surveillance cameras
SCFACE_IR_MUGSHOT = "cam8"            # the IR mugshot (see the distribution's mugshot_IR_cam8.txt)


def _scface_test():
    """{'001 · cam1 · d1': path} — everything under `surveillance_cameras_all/`, labelled by kind.

    That folder is not one population but three, and the label has to say which, because their
    accuracy differs by a factor of four (measured on all 130 subjects, ConvMAE-Base top-1):

        cam1-cam5, distances 1-3   visible surveillance, 1,950 imgs   43.5%   <- the report's figure
        cam6-cam7, distances 1-3   IR surveillance,        780 imgs   13.2%
        cam8                       IR *mugshot*,           130 imgs   63.9%

    Two file-naming traps live here. `cam8` has no distance token — it ships as `001_cam8.jpg`, one
    per subject — so an earlier `len(parts) >= 3` check dropped those 130 entries to the raw
    filename, leaving 4.5% of the picker reading `001_cam8.jpg` while the rest read
    `001 · cam1 · d1`, and any code parsing the label for a subject id skipped them silently. And
    `cam8` is a *mugshot* despite sitting in the surveillance folder. The subject is always
    `parts[0]`, and it stays first in the label so a parser only needs that.
    """
    root = scface_surveillance()
    if not root:
        return {}
    files = sorted(f for f in os.listdir(root) if f.lower().endswith(IMG_EXTS))
    out = {}
    for fn in files:
        parts = os.path.splitext(fn)[0].split("_")
        bits = [parts[0]] + parts[1:2] + [f"d{p}" for p in parts[2:3]]
        cam = parts[1] if len(parts) > 1 else ""
        if cam == SCFACE_IR_MUGSHOT:
            bits.append("IR mugshot")
        elif cam in SCFACE_IR_CAMS:
            bits.append("IR")
        out[" · ".join(bits) or fn] = os.path.join(root, fn)
    return out


def _scface_train_face(name):
    """The subject's high-res mugshot — the only thing the SCface fine-tune ever saw."""
    root = scface_mugshots()
    if not root:
        return None
    for fn in os.listdir(root):
        if fn.lower().endswith(IMG_EXTS) and fn.split("_")[0] == name:
            return os.path.join(root, fn)
    return None


# ------------------------------------------------------------------ LFW
def lfw_root():
    return _first_dir(os.path.join(DATASETS, "lfw", "lfw-deepfunneled", "lfw-deepfunneled"),
                      os.path.join(DATASETS, "lfw", "lfw-deepfunneled"),
                      os.path.join(DATASETS, "lfw", "lfw"))


def lfw_photo_counts():
    """{person: number of photos} from LFW's own people.csv — no directory listing needed."""
    def build():
        path = os.path.join(DATASETS, "lfw", "people.csv")
        if not os.path.isfile(path):
            return {}
        out = {}
        with open(path, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 2 and row[0].strip():
                    try:
                        out[row[0].strip()] = int(row[1])
                    except ValueError:
                        pass
        return out
    return _cached("lfw_counts", build)


def lfw_person_names(min_photos=1):
    """Every person in LFW (5,749), optionally only those with enough photos.

    Not restricted to the `pairs.csv` protocol: verification is open-set, so any two photos form a
    valid test, and filtering to the pairs list only shrank the choice to 4,281 for no benefit.

    `min_photos=2` is the filter that matters: someone with a single photo can only ever be paired
    against a *different* person, so they can never demonstrate a same-person match. LFW has 1,680
    such people — the same count as the fine-tune's LFW head.

    Counts come from people.csv rather than listing 5,749 directories, which cost ~0.4s at
    tab-build time for data only needed once a person is picked.
    """
    def build():
        root = lfw_root()
        return sorted(os.listdir(root)) if root else []

    names = _cached("lfw_names", build)
    if min_photos <= 1:
        return names
    counts = lfw_photo_counts()
    return [n for n in names if counts.get(n, len(lfw_person_photos(n))) >= min_photos]


def lfw_person_photos(person):
    """That one person's photo paths, listed on demand."""
    root = lfw_root()
    if not root or not person:
        return []
    return [os.path.join(root, person, f) for f in _identity_images(root, person)]


# ------------------------------------------------------------------ provenance of a dropped image
def _identity_index(job):
    """{filename: (split, identity)} for the identity tasks, built on first use.

    Filenames are unique within each of these datasets (CASIA names files by RecordIO key, SCface
    by subject+camera+distance), so a basename is enough to place a dropped file.
    """
    def build():
        trees = []
        if job == "CASIA":
            trees = [("train", casia_split("train")), ("val", casia_split("val"))]
        elif job == "SCface":
            trees = [("train", scface_mugshots()), ("val", scface_surveillance())]
        out = {}
        for split, root in trees:
            if not root:
                continue
            entries = sorted(os.listdir(root))
            if all(os.path.isfile(os.path.join(root, e)) for e in entries[:5]):
                for fn in entries:                      # flat directory (SCface)
                    if fn.lower().endswith(IMG_EXTS):
                        out[fn] = (split, fn.split("_")[0])
            else:
                for nm in entries:                      # per-identity folders (CASIA)
                    for fn in _identity_images(root, nm):
                        out[fn] = (split, nm)
        return out
    return _cached(f"locate:{job}", build)


def locate(job, filename):
    """Where a dropped image comes from: (split, identity), or None if it is not in the dataset."""
    if not filename:
        return None
    return _identity_index(job).get(os.path.basename(filename))


# ------------------------------------------------------------------ ImageNet (linear probe)
def imagenet_val_dir():
    """The ImageNet-1K validation image folder feeding the linear-probe tab's picker.

    Two layouts are accepted, because only one of them is worth downloading:

    * ``<wnid>/ILSVRC2012_val_*.JPEG`` — the val-only Kaggle mirror `titericz/imagenet1k-val`
      (~6.7 GB). The folder name IS the ground-truth class, so the tab can score the prediction.
    * a flat folder of 50,000 JPEGs — what the Kaggle *competition*
      `imagenet-object-localization-challenge` extracts to. Only reachable by pulling its ~155 GB
      archive, and it carries no per-image label here, so it is the fallback, not the default.

    Set ``IMAGENET_VAL_DIR`` to point at a copy you already have and skip the download entirely.
    """
    env = os.environ.get("IMAGENET_VAL_DIR")
    if env and os.path.isdir(env):
        return env
    return _first_dir(
        os.path.join(DATASETS, "imagenet", "imagenet-val"),                     # val-only mirror
        os.path.join(DATASETS, "imagenet", "val"),
        os.path.join(DATASETS, "imagenet", "ILSVRC", "Data", "CLS-LOC", "val"),  # competition
        os.path.join(DATASETS, "imagenet"))


def _imagenet_test():
    """{label: path} over the 50,000 held-out val images — the probe trained on `train/` only.

    Labelled as ``<wnid>/<file>`` when the tree is class-foldered, so the picker entry itself says
    which class the image really is; a flat tree falls back to the bare filename.
    """
    root = imagenet_val_dir()
    if not root:
        return {}
    entries = sorted(os.listdir(root))
    wnids = [e for e in entries
             if e.startswith("n") and os.path.isdir(os.path.join(root, e))]
    if wnids:
        return {f"{w}/{fn}": os.path.join(root, w, fn)
                for w in wnids for fn in _identity_images(root, w)}
    return {fn: os.path.join(root, fn) for fn in entries if fn.lower().endswith(IMG_EXTS)}


def _imagenet_index():
    """The bundled imagenet_class_index.json, as {"0": ["n01440764", "tench"], ...}."""
    def build():
        path = os.path.join(ASSETS, "imagenet_class_index.json")
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)
    return _cached("imagenet_index", build)


def imagenet_class_names():
    """{class_index: human name} from the bundled imagenet_class_index.json (sorted-wnid order,
    the ImageFolder convention the probe trained under)."""
    return _cached("imagenet_class_names",
                   lambda: {int(k): v[1].replace("_", " ") for k, v in _imagenet_index().items()})


def imagenet_truth(rel_or_path):
    """(class index, name) for a val image, read from the wnid folder it sits in; None if absent.

    The class index is the position in sorted-wnid order — exactly what `ImageFolder` handed the
    probe at training time — so the folder name alone is a sound ground truth, no label file needed.
    """
    if not rel_or_path:
        return None
    wnid = os.path.basename(os.path.dirname(str(rel_or_path).replace("\\", "/")))
    idx = _cached("imagenet_wnids", lambda: {v[0]: int(k) for k, v in _imagenet_index().items()})
    i = idx.get(wnid)
    return None if i is None else (i, imagenet_class_names().get(i, wnid))


# ------------------------------------------------------------------ public API
_TEST = {"CelebA": _celeba_test, "CASIA": _casia_test, "SCface": _scface_test,
         "ImageNet": _imagenet_test}


def test_samples(job):
    """Held-out images offered in the tab's dropdown: {label: absolute path}.

    LFW is absent on purpose — verification is open-set, so no LFW identity is a training class
    and any two photos form a valid test. That tab picks person → photo instead.
    """
    return _cached(f"test:{job}", _TEST.get(job, dict))


def class_names(job):
    """Identity folder names in sorted order — CASIA's picker and the class_index cache use this.

    NOT a class-index lookup for CASIA: use class_label()/class_face() for that.
    """
    if job == "SCface":
        return [f"{i:03d}" for i in range(1, SCFACE_SUBJECTS + 1)]
    if job == "CASIA":
        return _cached("casia_classes", _casia_classes)
    return []


def class_label(job, index):
    """Human-readable identity for a predicted class index, or None if it cannot be named."""
    if job == "SCface":
        return f"{index + 1:03d}" if 0 <= index < SCFACE_SUBJECTS else None
    if job == "CASIA":
        names = class_names("CASIA")
        return names[index] if 0 <= index < len(names) else None
    if job == "ImageNet":
        return imagenet_class_names().get(index)
    return None


def class_face(job, index):
    """A TRAIN image of that class, for the predicted-person panel; None if unavailable."""
    name = class_label(job, index)
    if not name:
        return None
    if job == "SCface":
        return _scface_train_face(name)
    if job == "CASIA":
        return _casia_train_face(name)
    return None


def status(job):
    """One-line description of what the picker is offering, shown under each dropdown."""
    n = len(test_samples(job))
    if not n:
        return "Dataset not found — run `python tools/fetch_datasets.py`, or drop your own image below."
    if job == "CASIA":
        have, total = len(casia_extracted()), len(class_names("CASIA"))
        extra = ""
        if have < total:
            extra = (f" Only {have:,}/{total:,} identities are on disk — re-run "
                     "`python tools/extract_casia_recordio.py` (it resumes) for the rest.")
        return (f"{n:,} held-out images ({CASIA_VAL_PER_CLASS}/identity) across {have:,} "
                f"identities — none of these were trained on.{extra}")
    if job == "ImageNet":
        labelled = " Each entry is prefixed with its ground-truth wnid." if "/" in next(
            iter(test_samples(job))) else " Flat tree — no per-image ground truth available."
        return (f"{n:,} held-out ImageNet-1K val images — the probe never trained on any of "
                f"them.{labelled}")
    if job == "SCface":
        return (f"{n:,} held-out images — the model only trained on the visible mugshots. Three "
                "kinds, and they behave very differently: 1,950 visible surveillance crops "
                "(cam1-5, the report's setting), 780 marked **IR** (cam6-7, far harder), and 130 "
                "marked **IR mugshot** (cam8).")
    return (f"{n:,} samples from the CelebA val+test partitions (1 & 2) — none of these were "
            "trained on.")
