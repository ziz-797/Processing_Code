import csv
import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

BASE_DIR  = "./VinDr-CXR"
CSV_TRAIN = os.path.join(BASE_DIR, "annotations_train.csv")
CSV_TEST  = os.path.join(BASE_DIR, "annotations_test.csv")
IMG_DIR   = os.path.join(BASE_DIR, "VinDr-CXR-dataset", "ai-vinbigdata")
CROP_DIR  = os.path.join(BASE_DIR, "crop")

OUTPUT_TRAIN_JSON = os.path.join(BASE_DIR, "VinDr-CXR_train.json")
OUTPUT_TEST_JSON  = os.path.join(BASE_DIR, "VinDr-CXR_test.json")

MIN_SIZE         = 28
IOU_THR          = 0.5
CONTAIN_THR      = 0.7
NUM_WORKERS      = 8
MAX_TEST_SAMPLES = 1000
RANDOM_SEED      = 42


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

def intersect_area(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)

def should_merge(a, b):
    inter = intersect_area(a, b)
    if inter == 0:
        return False
    iou     = inter / (area(a) + area(b) - inter)
    contain = inter / min(area(a), area(b))
    return iou >= IOU_THR or contain >= CONTAIN_THR

def union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))

def merge_boxes(boxes):
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if should_merge(boxes[i], boxes[j]):
                    merged = union_box([boxes[i], boxes[j]])
                    boxes = [boxes[k] for k in range(len(boxes)) if k not in (i, j)]
                    boxes.append(merged)
                    changed = True
                    break
            if changed:
                break
    return boxes


def crop_with_min_size(img, x_min, y_min, x_max, y_max, min_size=MIN_SIZE):
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    w  = max(x_max - x_min, min_size)
    h  = max(y_max - y_min, min_size)
    x_min = max(0, cx - w // 2); x_max = min(img.width,  x_min + w); x_min = max(0, x_max - w)
    y_min = max(0, cy - h // 2); y_max = min(img.height, y_min + h); y_min = max(0, y_max - h)
    return img.crop((x_min, y_min, x_max, y_max))


def normalize_path(path):
    return str(path).replace("\\", "/")

def remove_root_prefix(path, root):
    path = normalize_path(path)
    root = normalize_path(root).rstrip("/")
    if path.startswith(root + "/"):
        return path[len(root) + 1:]
    elif path == root:
        return ""
    return path


def process_group(args):
    key, before_boxes, class_id, split = args
    image_id, class_name = key

    img_subdir = "train" if split == "training" else "test"
    img_path   = os.path.join(IMG_DIR, img_subdir, image_id + ".png")

    try:
        img = Image.open(img_path)
    except Exception as e:
        return None, f"open {image_id}: {e}"

    after_boxes = merge_boxes(before_boxes)

    crop_paths = []
    for idx, box in enumerate(after_boxes):
        crop_name = f"{image_id}_{class_id}_{idx}.png"
        crop_path = os.path.join(CROP_DIR, crop_name)
        try:
            crop = crop_with_min_size(img, *box)
            crop.save(crop_path)
            crop_paths.append(crop_path)
        except Exception as e:
            return None, f"crop {image_id} box{idx}: {e}"

    if not crop_paths:
        return None, None

    return {
        "image_path":       img_path,
        "class_name":       class_name,
        "split":            split,
        "crop_image_paths": crop_paths,
    }, None


def convert_record(record):
    img_path_rel   = remove_root_prefix(record["image_path"], BASE_DIR)
    crop_paths_rel = [remove_root_prefix(p, BASE_DIR) for p in record["crop_image_paths"]]
    tgt_img_path   = crop_paths_rel if len(crop_paths_rel) <= 1 else [crop_paths_rel]

    return {
        "qry_inst":     "<|image_1|> Locate the specific region that corresponds to the provided text description.",
        "qry_text":     record["class_name"],
        "qry_img_path": img_path_rel,
        "tgt_inst":     "Match the target",
        "tgt_text":     ["<|image_1|>\n"],
        "tgt_img_path": tgt_img_path,
    }


def sample_test_data(data, max_samples, seed=RANDOM_SEED):
    if len(data) <= max_samples:
        return data

    rng = random.Random(seed)
    path_to_samples = {}
    for s in data:
        path_to_samples.setdefault(s["qry_img_path"], []).append(s)

    unique_paths = list(path_to_samples.keys())
    rng.shuffle(unique_paths)

    sampled = []
    for path in unique_paths:
        sampled.append(rng.choice(path_to_samples[path]))
        if len(sampled) >= max_samples:
            break

    if len(sampled) < max_samples:
        selected_ids = {id(x) for x in sampled}
        remaining    = [x for x in data if id(x) not in selected_ids]
        rng.shuffle(remaining)
        sampled.extend(remaining[:max_samples - len(sampled)])

    rng.shuffle(sampled)
    return sampled


def main():
    os.makedirs(CROP_DIR, exist_ok=True)

    with open(CSV_TRAIN, newline="") as f:
        train_rows = list(csv.DictReader(f))
    with open(CSV_TEST, newline="") as f:
        test_rows = list(csv.DictReader(f))
    print(f"Raw rows: train={len(train_rows)}  test={len(test_rows)}")

    groups = defaultdict(list)
    img_split = {}
    all_classes = set()

    for split_name, rows in [("training", train_rows), ("test", test_rows)]:
        for row in rows:
            if row["class_name"] == "No finding" or not row.get("x_min", "").strip():
                continue
            key = (row["image_id"], row["class_name"])
            groups[key].append((
                max(0, int(float(row["x_min"]))),
                max(0, int(float(row["y_min"]))),
                max(0, int(float(row["x_max"]))),
                max(0, int(float(row["y_max"]))),
            ))
            img_split[row["image_id"]] = split_name
            all_classes.add(row["class_name"])

    class_to_id = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
    print(f"Classes ({len(class_to_id)}): {class_to_id}")

    tasks = [
        (key, boxes, class_to_id[key[1]], img_split[key[0]])
        for key, boxes in groups.items()
    ]
    print(f"Total groups: {len(tasks)}  Workers: {NUM_WORKERS}")

    train_records = []
    test_records  = []
    errors = 0

    with Pool(NUM_WORKERS) as pool:
        for i, (record, err) in enumerate(
            pool.imap_unordered(process_group, tasks, chunksize=16), start=1
        ):
            if err:
                print(f"  ERROR: {err}")
                errors += 1
            elif record:
                entry = convert_record(record)
                if record["split"] == "training":
                    train_records.append(entry)
                else:
                    test_records.append(entry)

            if i % 500 == 0:
                print(f"  [{i}/{len(tasks)}] errors={errors}")

    print(f"\nProcessing complete. errors={errors}")
    print(f"  training samples              : {len(train_records)}")
    print(f"  test samples (before sampling): {len(test_records)}")

    test_records = sample_test_data(test_records, MAX_TEST_SAMPLES, RANDOM_SEED)
    print(f"  test samples (after  sampling): {len(test_records)}")

    for path, data in [(OUTPUT_TRAIN_JSON, train_records), (OUTPUT_TEST_JSON, test_records)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
