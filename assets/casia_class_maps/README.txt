CASIA class-index -> identity ordering  (for the demo "predicted person" feature)
=================================================================================
Problem: the demo had no record of the ImageFolder class ordering used at fine-tune
time, so a rebuilt list was in the wrong order (1% train-acc = order fully wrong).

What these files are
--------------------
casia_classes_baseline.json / casia_classes_ghost.json
    The exact class ordering = ImageFolder(train_dir).classes = sorted(os.listdir(train_dir))
    of the prepare_casia.py output used for fine-tuning. 10,571 entries.
    classes[i] is the folder for class index i. Both arms are byte-identical (see md5).

    Folder names are the CASIA-WebFace RecordIO integer labels, zero-padded to 5 digits
    ("00000".."10571"). They are NOT the original CASIA person-id strings.

Key detail (why naive reconstruction fails)
--------------------------------------------
The labels are NOT contiguous: label 09282 has no decodable image, so it is absent.
  - class index i -> folder classes[i]
  - for i <  9282: classes[i] == f"{i:05d}"
  - for i >= 9282: classes[i] == f"{i+1:05d}"   (shifted by the single gap)
  e.g. class index 9282 -> folder "09283";  folder "10571" -> class index 10570.
Any reconstruction that assumes contiguous 0..N, or that sorts the ORIGINAL CASIA
person-id folders, gives the wrong order.

How to use (deploy side)
------------------------
  classes = json.load(open("casia_classes_baseline.json"))   # or _ghost
  pred_index = int(logits.argmax())          # 0..10570
  identity_folder = classes[pred_index]       # e.g. "00042"
  # show a face: any image under  <prepared_casia>/train/<identity_folder>/*.jpg

Why one list serves both arms
-----------------------------
Both checkpoints have head [10571, 768]. prepare_casia.py is deterministic: it creates
one folder f"{label:05d}" per RecordIO label that has >=1 decodable image, so the ordering
depends only on the raw CASIA RecordIO + prepare params, not on the output path name
(baseline cfg=data/casia_full, ghost cfg=data/casia were both the full 10,571-class prepare
of the same raw dump). The lists are therefore identical (md5 above).
Caveat: if the ghost run had used a *different* raw dump/params, the order could differ;
the matching 10,571 head count makes that very unlikely. If you want certainty, run
prepare_casia.py on the raw dump and diff sorted(os.listdir(out/train)) against these files.
