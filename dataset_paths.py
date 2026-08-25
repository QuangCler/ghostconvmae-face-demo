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

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(HERE, "datasets")

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

    `datasets/` is not in version control, so the copy under advice/ is the one a fresh clone gets;
    reading it directly keeps the CASIA tab working without the dataset having to be present.
    """
    for path in (os.path.join(DATASETS, "casia", "class_order.json"),
                 os.path.join(HERE, "advice", "casia_class_maps", "casia_classes_baseline.json")):
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


def _scface_test():
    """{'001 · cam1 · d1': path} — surveillance crops, named `<subject>_cam<n>_<distance>.jpg`."""
    root = scface_surveillance()
    if not root:
        return {}
    files = sorted(f for f in os.listdir(root) if f.lower().endswith(IMG_EXTS))
    out = {}
    for fn in files:
        parts = os.path.splitext(fn)[0].split("_")
        label = f"{parts[0]} · {parts[1]} · d{parts[2]}" if len(parts) >= 3 else fn
        out[label] = os.path.join(root, fn)
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


# ------------------------------------------------------------------ public API
_TEST = {"CelebA": _celeba_test, "CASIA": _casia_test, "SCface": _scface_test}


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
        return "Dataset not found — run `python fetch_datasets.py`, or drop your own image below."
    if job == "CASIA":
        have, total = len(casia_extracted()), len(class_names("CASIA"))
        extra = ""
        if have < total:
            extra = (f" Only {have:,}/{total:,} identities are on disk — re-run "
                     "`python extract_casia_recordio.py` (it resumes) for the rest.")
        return (f"{n:,} held-out images ({CASIA_VAL_PER_CLASS}/identity) across {have:,} "
                f"identities — none of these were trained on.{extra}")
    src = {"CelebA": "CelebA val+test partitions (1 & 2)",
           "SCface": "surveillance crops (the model only trained on mugshots)"}[job]
    return f"{n:,} samples from the {src} — none of these were trained on."
