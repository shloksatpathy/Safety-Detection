"""
Train a YOLOv8n model on the Hard Hat Workers dataset.

Run with:
    python training.py
"""

# =====================================================================================
# STEP 0 -- WHAT'S ACTUALLY IN data/archive/ (read this before touching the code below)
# =====================================================================================
# data/archive/images/       5000 PNG images, named hard_hat_workers<id>.png,
#                             each ~416x416 (a handful are 415x416 or 416x415 --
#                             i.e. NOT all identical, so always read real width/height
#                             per image instead of hardcoding 416).
#
# data/archive/annotations/  5000 Pascal VOC XML files, one per image, named
#                             hard_hat_workers<id>.xml. Example structure:
#
#                                 <annotation>
#                                   <filename>hard_hat_workers4479.png</filename>
#                                   <size><width>416</width><height>415</height></size>
#                                   <object>
#                                     <name>helmet</name>
#                                     <bndbox>
#                                       <xmin>85</xmin><ymin>74</ymin>
#                                       <xmax>191</xmax><ymax>200</ymax>
#                                     </bndbox>
#                                   </object>
#                                   ... (0 or more <object> per image)
#                                 </annotation>
#
# Classes present across all 5000 files (counted directly from the XML <name> tags):
#     helmet -> 18966 boxes   head -> 5785 boxes   person -> 751 boxes
#
# PROBLEM: ultralytics/YOLO does not understand VOC XML. It expects, per image, a
# .txt file with one line per box:
#
#     <class_id> <x_center> <y_center> <width> <height>
#
# where all four numbers are normalized to [0, 1] by the image's width/height (not
# raw pixels), and class_id is an integer index into a fixed class list. It also
# expects a specific folder layout:
#
#     <dataset>/images/train/*.png       <dataset>/labels/train/*.txt
#     <dataset>/images/val/*.png         <dataset>/labels/val/*.txt
#
# plus a data.yaml that says where train/val live and what the class names are.
#
# None of that exists yet -- there's no data.yaml, no labels/ folder, no train/val
# split anywhere in data/. So this script's job, in order, is:
#
#   1. Parse every VOC XML file and convert its pixel boxes to YOLO-normalized txt.
#   2. Randomly split the images into train/val and copy each image + its new label
#      into the images/{train,val} + labels/{train,val} layout above.
#   3. Write data.yaml describing that layout and the 3 class names.
#   4. Load the yolov8n.pt checkpoint via ultralytics and call model.train().
#
# NOTE: `ultralytics` is NOT currently installed (checked with `pip list`) and is
# NOT in requirements.txt. Install it before running training:
#     pip install ultralytics
# =====================================================================================

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# --------------------------------------------------------------------------------
# CONFIG -- the knobs you're most likely to want to change
# --------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "data" / "archive" / "images"
ANNOTATIONS_DIR = ROOT / "data" / "archive" / "annotations"
DATASET_DIR = ROOT / "data" / "yolo_dataset"  # converted train/val dataset gets written here

# Order matters here: CLASS_NAMES[i] is what ultralytics calls class_id i, and the
# id we write into every label .txt file must match this same order.
CLASS_NAMES = ["helmet", "head", "person"]

VAL_SPLIT = 0.2  # fraction of images held out for validation (0.2 = 1000 of 5000)
SEED = 42  # fixes the train/val split and training RNG so runs are reproducible

EPOCHS = 50
IMG_SIZE = 416  # matches the dataset's native resolution, so no big up/downscale
BATCH_SIZE = 16
MODEL_WEIGHTS = "yolov8n.pt"  # nano checkpoint; ultralytics auto-downloads it on first use

# Ultralytics otherwise writes runs to the global `runs_dir` in
# ~/.config/Ultralytics/settings.json, which points at whatever project set it last.
# Pin it here so results always land next to this script.
RUNS_DIR = ROOT / "runs" / "detect"
RUN_NAME = "train"


def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Convert one VOC box, given in absolute pixel corners (xmin, ymin, xmax, ymax),
    into YOLO's normalized (x_center, y_center, width, height) format, each a
    fraction of the image width/height in [0, 1].
    """
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def convert_annotations():
    """
    Read every VOC XML file in ANNOTATIONS_DIR, convert its boxes to YOLO lines,
    and write one .txt per image into a scratch "labels_all" folder (this gets
    split into train/val and cleaned up by split_and_layout()).

    Returns the list of image stems (filename without extension) that ended up
    with at least one recognized box -- these are the only images we train on.
    """
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    labels_tmp_dir = DATASET_DIR / "labels_all"
    labels_tmp_dir.mkdir(parents=True, exist_ok=True)

    converted_stems = []
    for xml_path in sorted(ANNOTATIONS_DIR.glob("*.xml")):
        tree = ET.parse(xml_path)
        xml_root = tree.getroot()

        size = xml_root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        lines = []
        for obj in xml_root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name not in class_to_id:
                # Guard against a stray/typo'd class label instead of crashing the
                # whole conversion run over one bad file.
                continue

            box = obj.find("bndbox")
            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)

            xc, yc, w, h = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            lines.append(f"{class_to_id[name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not lines:
            continue  # no recognized boxes -- skip this image entirely

        stem = xml_path.stem  # e.g. "hard_hat_workers4479"
        (labels_tmp_dir / f"{stem}.txt").write_text("\n".join(lines))
        converted_stems.append(stem)

    return converted_stems


def split_and_layout(stems):
    """
    Shuffle the given image stems with a fixed seed, split them into train/val by
    VAL_SPLIT, and copy each image + its converted label into the
    images/{train,val} and labels/{train,val} folders ultralytics expects.
    """
    random.seed(SEED)
    stems = stems[:]
    random.shuffle(stems)

    n_val = int(len(stems) * VAL_SPLIT)
    splits = {"val": stems[:n_val], "train": stems[n_val:]}

    labels_tmp_dir = DATASET_DIR / "labels_all"

    for split_name, split_stems in splits.items():
        img_out = DATASET_DIR / "images" / split_name
        lbl_out = DATASET_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for stem in split_stems:
            src_img = IMAGES_DIR / f"{stem}.png"
            src_lbl = labels_tmp_dir / f"{stem}.txt"
            shutil.copy(src_img, img_out / src_img.name)
            shutil.copy(src_lbl, lbl_out / src_lbl.name)

    shutil.rmtree(labels_tmp_dir)  # scratch folder no longer needed once split
    return {name: len(s) for name, s in splits.items()}


def write_data_yaml():
    """
    Write the data.yaml ultralytics reads to find images/labels and to map
    class_id -> class name. The `names` order here must match CLASS_NAMES, since
    that's the order convert_annotations() used to assign ids.
    """
    yaml_path = DATASET_DIR / "data.yaml"
    lines = [
        f"path: {DATASET_DIR}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    yaml_path.write_text("\n".join(lines))
    return yaml_path


def train():
    """
    Load a YOLOv8n checkpoint and fine-tune it on our converted dataset.
    Imported lazily so the conversion/split steps above still work even before
    `pip install ultralytics` has been run.
    """
    from ultralytics import YOLO

    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        seed=SEED,
        project=str(RUNS_DIR),
        name=RUN_NAME,
    )


if __name__ == "__main__":
    # Only redo the VOC->YOLO conversion + split if we haven't already built the
    # dataset -- re-running it every time would just recopy 5000 images for nothing.
    if not (DATASET_DIR / "data.yaml").exists():
        print("Converting VOC annotations to YOLO format...")
        image_stems = convert_annotations()
        split_counts = split_and_layout(image_stems)
        write_data_yaml()
        print(f"Dataset ready at {DATASET_DIR}: {split_counts}")
    else:
        print(f"Found existing dataset at {DATASET_DIR} -- skipping conversion.")

    train()
