import csv
import json
import os
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Pool

from PIL import Image

BASE_DIR  = "./DeepLesion"
CSV_PATH  = os.path.join(BASE_DIR, "DL_info.csv")
RAW_DIR   = os.path.join(BASE_DIR, "image")
WN_DIR    = os.path.join(BASE_DIR, "images_windowed")
CROP_DIR  = os.path.join(BASE_DIR, "crop")

OUTPUT_TRAIN_JSON = os.path.join(BASE_DIR, "DeepLesion_train.json")
OUTPUT_TEST_JSON  = os.path.join(BASE_DIR, "DeepLesion_test.json")

MIN_SIZE        = 28
NUM_WORKERS     = 16
TEST_SAMPLE_SIZE = 1000
RANDOM_SEED     = 42

SPLIT_MAP = {"1": "training", "2": "validation", "3": "test"}
TYPE_MAP  = {
    "1": "bone", "2": "abdomen", "3": "mediastinum",
    "4": "liver", "5": "lung", "6": "kidney",
    "7": "soft_tissue", "8": "pelvis", "-1": "unknown",
}


def crop_with_min_size(img, x_min, y_min, x_max, y_max, min_size=MIN_SIZE):
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    w  = max(x_max - x_min, min_size)
    h  = max(y_max - y_min, min_size)
    x_min = max(0, cx - w // 2); x_max = min(img.width,  x_min + w); x_min = max(0, x_max - w)
    y_min = max(0, cy - h // 2); y_max = min(img.height, y_min + h); y_min = max(0, y_max - h)
    return img.crop((x_min, y_min, x_max, y_max))


def normalize_path(path):
    return str(path).replace("\\", "/").strip()

def remove_root_prefix(path, root):
    path = normalize_path(path)
    root = normalize_path(root).rstrip("/")
    if path == root:
        return ""
    if path.startswith(root + "/"):
        return path[len(root) + 1:]
    return path


def process_lesion(args):
    file_name, dup_idx, bbox, dicom_wins, lesion_type, split = args
    folder    = file_name.rsplit("_", 1)[0]
    key_slice = file_name.rsplit("_", 1)[1]
    raw_path  = os.path.join(RAW_DIR, folder, key_slice)
    stem      = file_name.replace(".png", "")
    out_name  = f"{stem}-{dup_idx}.png"
    crop_path = os.path.join(CROP_DIR, out_name)

    try:
        raw = np.array(Image.open(raw_path))
        hu  = raw.astype("int32") - 32768
        clip_min, clip_max = dicom_wins
        wn  = np.clip(hu, clip_min, clip_max)
        wn  = ((wn - clip_min) / (clip_max - clip_min) * 255).astype("uint8")
        img = Image.fromarray(wn, mode="L")
        wn_folder = os.path.join(WN_DIR, folder)
        os.makedirs(wn_folder, exist_ok=True)
        wn_path = os.path.join(wn_folder, key_slice)
        if not os.path.exists(wn_path):
            img.save(wn_path)
    except Exception as e:
        return None, f"window {file_name}: {e}"

    x1, y1, x2, y2 = bbox
    try:
        crop_with_min_size(img, x1, y1, x2, y2).save(crop_path)
    except Exception as e:
        return None, f"crop {out_name}: {e}"

    return {
        "image_path":       raw_path,
        "class_name":       TYPE_MAP.get(lesion_type, "unknown"),
        "split":            split,
        "crop_image_paths": [crop_path],
    }, None


def convert_record(record):
    image_path_rel = remove_root_prefix(record["image_path"], BASE_DIR)
    crop_rel       = remove_root_prefix(record["crop_image_paths"][0], BASE_DIR)
    return {
        "qry_inst":     "<|image_1|> Locate the lesion in the image:",
        "qry_text":     "",
        "qry_img_path": image_path_rel,
        "tgt_inst":     "Match the target",
        "tgt_text":     ["<|image_1|>\n"],
        "tgt_img_path": [crop_rel],
    }


def main():
    os.makedirs(WN_DIR,   exist_ok=True)
    os.makedirs(CROP_DIR, exist_ok=True)

    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Total lesion rows: {len(rows)}")

    fname_counter = defaultdict(int)
    tasks = []
    for row in rows:
        file_name = row["File_name"]
        dup_idx   = fname_counter[file_name]
        fname_counter[file_name] += 1

        bbox_vals  = [float(x.strip()) for x in row["Bounding_boxes"].split(",")]
        bbox       = (int(bbox_vals[0]), int(bbox_vals[1]), int(bbox_vals[2]), int(bbox_vals[3]))
        dicom_wins = tuple(float(v.strip()) for v in row["DICOM_windows"].split(","))
        split      = SPLIT_MAP.get(row["Train_Val_Test"], "unknown")
        tasks.append((file_name, dup_idx, bbox, dicom_wins, row["Coarse_lesion_type"], split))

    print(f"Tasks: {len(tasks)}  workers: {NUM_WORKERS}")

    train_records = []
    test_records  = []
    errors        = 0

    with Pool(NUM_WORKERS) as pool:
        for i, (record, err) in enumerate(
            pool.imap_unordered(process_lesion, tasks, chunksize=64), start=1
        ):
            if err:
                print(f"  ERROR: {err}"); errors += 1
            elif record:
                if record["split"] == "training":
                    train_records.append(record)
                elif record["split"] == "test":
                    test_records.append(record)
            if i % 5000 == 0:
                print(f"  [{i}/{len(tasks)}] errors={errors}")

    print(f"\nProcessing complete. errors={errors}")
    print(f"  training: {len(train_records)}  test: {len(test_records)}")

    train_seen = set()
    train_data = []
    for r in train_records:
        entry = convert_record(r)
        key   = (entry["qry_img_path"], entry["tgt_img_path"][0])
        if key not in train_seen:
            train_seen.add(key)
            train_data.append(entry)

    test_seen = set()
    test_candidates = []
    for r in test_records:
        entry = convert_record(r)
        if entry["qry_img_path"] not in test_seen:
            test_seen.add(entry["qry_img_path"])
            test_candidates.append(entry)

    rng = random.Random(RANDOM_SEED)
    test_data = rng.sample(test_candidates, min(TEST_SAMPLE_SIZE, len(test_candidates)))

    print(f"  train (dedup): {len(train_data)}")
    print(f"  test (unique img, before sampling): {len(test_candidates)}")
    print(f"  test (after sampling): {len(test_data)}")

    for path, data in [(OUTPUT_TRAIN_JSON, train_data), (OUTPUT_TEST_JSON, test_data)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
