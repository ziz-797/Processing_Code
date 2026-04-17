import csv
import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool

from PIL import Image

BASE_DIR    = "./Chest_imagenome"
DATA_DIR    = os.path.join(BASE_DIR, "gold")
BBOX_CSV    = os.path.join(DATA_DIR, "gold_bbox_coordinate_annotations_1000images.csv")
SENTENCES   = os.path.join(DATA_DIR, "gold_all_sentences_500pts_1000studies.txt")
SILVER_DIR  = os.path.join(BASE_DIR, "silver")
SG_DIR      = os.path.join(SILVER_DIR, "scene_graph")
SPLIT_DIR   = os.path.join(SILVER_DIR, "split")
MIMIC_DIR   = "./mimic_original/2.0.0/files"

GOLD_CROP_DIR   = os.path.join(BASE_DIR, "gold_crop")
SILVER_CROP_DIR = os.path.join(BASE_DIR, "silver_sample_crop")

OUTPUT_TRAIN_JSON = os.path.join(BASE_DIR, "Chest_imagenome_train.json")
OUTPUT_TEST_JSON  = os.path.join(BASE_DIR, "Chest_imagenome_test.json")

TARGET_TOTAL     = 1_000_000
MIN_SIZE         = 28
NUM_WORKERS_GOLD   = 8
NUM_WORKERS_SILVER = 16
MAX_TEST_SAMPLES = 1000
RANDOM_SEED      = 42
SEED             = 42

TRAIN_IMAGE_ROOT = MIMIC_DIR
TRAIN_CROP_ROOT  = BASE_DIR
TEST_IMAGE_ROOT  = MIMIC_DIR
TEST_CROP_ROOT   = BASE_DIR


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


def process_gold_image(args):
    image_id, regions, class_to_id, img_path = args

    try:
        img = Image.open(img_path)
    except Exception as e:
        return None, f"open {image_id}: {e}"

    map_rows = []
    for bbox_name, box in regions:
        class_id  = class_to_id[bbox_name]
        crop_path = os.path.join(GOLD_CROP_DIR, f"{image_id}_{class_id}_0.png")
        try:
            crop_with_min_size(img, *box).save(crop_path)
        except Exception as e:
            return None, f"crop {image_id} {bbox_name}: {e}"

        map_rows.append({
            "image_path":       img_path,
            "class_name":       bbox_name,
            "crop_image_paths": [crop_path],
        })

    return map_rows, None


def run_gold():
    os.makedirs(GOLD_CROP_DIR, exist_ok=True)

    img_to_path = {}
    with open(SENTENCES, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            image_id = row["image_id"]
            if image_id in img_to_path:
                continue
            patient_id = row["patient_id"]
            study_id   = row["study_id"]
            jpg_name   = image_id.replace(".dcm", ".jpg")
            img_to_path[image_id] = os.path.join(
                MIMIC_DIR, f"p{patient_id[:2]}", f"p{patient_id}", f"s{study_id}", jpg_name
            )
    print(f"[gold] image path mapping: {len(img_to_path)}")

    image_regions = defaultdict(list)
    all_names = set()
    with open(BBOX_CSV, newline="") as f:
        for row in csv.DictReader(f):
            image_id  = row["image_id"]
            bbox_name = row["bbox_name"]
            box = (
                max(0, int(float(row["original_x1"]))),
                max(0, int(float(row["original_y1"]))),
                max(0, int(float(row["original_x2"]))),
                max(0, int(float(row["original_y2"]))),
            )
            image_regions[image_id].append((bbox_name, box))
            all_names.add(bbox_name)

    class_to_id = {name: idx for idx, name in enumerate(sorted(all_names))}
    print(f"[gold] regions: {len(class_to_id)}  images: {len(image_regions)}")

    tasks = []
    for image_id, regions in image_regions.items():
        if image_id not in img_to_path:
            continue
        img_id_short = image_id.replace(".dcm", "")
        tasks.append((img_id_short, regions, class_to_id, img_to_path[image_id]))

    print(f"[gold] tasks: {len(tasks)}  workers: {NUM_WORKERS_GOLD}")

    records = []
    errors  = 0
    with Pool(NUM_WORKERS_GOLD) as pool:
        for i, (map_rows, err) in enumerate(
            pool.imap_unordered(process_gold_image, tasks, chunksize=4), start=1
        ):
            if err:
                print(f"  ERROR: {err}"); errors += 1
            elif map_rows:
                records.extend(map_rows)
            if i % 200 == 0:
                print(f"  [gold {i}/{len(tasks)}] errors={errors}")

    print(f"[gold] done. records={len(records)} errors={errors}")
    return records


def scan_one_json(args):
    sg_path, dicom_id = args
    try:
        with open(sg_path) as f:
            sg = json.load(f)
    except Exception:
        return None
    names = [obj["bbox_name"] for obj in sg.get("objects", []) if obj.get("original_x1") is not None]
    return (dicom_id, names) if names else None


def process_silver_image(args):
    dicom_id, img_path, sg_path, split, selected_names, class_to_id = args

    try:
        with open(sg_path) as f:
            sg = json.load(f)
    except Exception as e:
        return None, f"json {dicom_id}: {e}"

    obj_map = {obj["bbox_name"]: obj for obj in sg.get("objects", [])}

    try:
        img = Image.open(img_path)
    except Exception as e:
        return None, f"open {dicom_id}: {e}"

    map_rows = []
    for bbox_name in selected_names:
        obj = obj_map.get(bbox_name)
        if obj is None:
            continue
        ox1, oy1, ox2, oy2 = obj.get("original_x1"), obj.get("original_y1"), obj.get("original_x2"), obj.get("original_y2")
        if any(v is None for v in (ox1, oy1, ox2, oy2)):
            continue

        box      = (max(0, int(ox1)), max(0, int(oy1)), max(0, int(ox2)), max(0, int(oy2)))
        class_id = class_to_id[bbox_name]
        crop_path = os.path.join(SILVER_CROP_DIR, f"{dicom_id}_{class_id}_0.png")

        try:
            crop_with_min_size(img, *box).save(crop_path)
        except Exception as e:
            return None, f"crop {dicom_id} {bbox_name}: {e}"

        map_rows.append({
            "image_path":       img_path,
            "class_name":       bbox_name,
            "split":            split,
            "crop_image_paths": [crop_path],
        })

    return map_rows, None


def run_silver():
    os.makedirs(SILVER_CROP_DIR, exist_ok=True)
    random.seed(SEED)

    id_map = {}
    for split_name, csv_name in [("training", "train.csv"), ("validation", "valid.csv"), ("test", "test.csv")]:
        with open(os.path.join(SPLIT_DIR, csv_name), newline="") as f:
            for row in csv.DictReader(f):
                id_map[row["dicom_id"]] = (row["subject_id"], row["study_id"], split_name)
    print(f"[silver] split mapping: {len(id_map)}")

    sg_files  = [fn for fn in os.listdir(SG_DIR) if fn.endswith("_SceneGraph.json")]
    scan_args = [
        (os.path.join(SG_DIR, fn), fn.replace("_SceneGraph.json", ""))
        for fn in sg_files
        if fn.replace("_SceneGraph.json", "") in id_map
    ]
    print(f"[silver] scanning {len(scan_args)} JSONs...")

    anatomy_to_images = defaultdict(list)
    with Pool(NUM_WORKERS_SILVER) as pool:
        for i, result in enumerate(pool.imap_unordered(scan_one_json, scan_args, chunksize=256)):
            if result:
                dicom_id, names = result
                for name in names:
                    anatomy_to_images[name].append(dicom_id)
            if (i + 1) % 50000 == 0:
                print(f"  scanned {i+1}/{len(scan_args)}")

    n_anatomy  = len(anatomy_to_images)
    per_anatomy = TARGET_TOTAL // n_anatomy
    print(f"[silver] anatomy types: {n_anatomy}  per anatomy: {per_anatomy}")

    image_selected = defaultdict(set)
    for bbox_name, img_list in sorted(anatomy_to_images.items()):
        n_sample = min(per_anatomy, len(img_list))
        for dicom_id in random.sample(img_list, n_sample):
            image_selected[dicom_id].add(bbox_name)

    class_to_id = {name: idx for idx, name in enumerate(sorted(anatomy_to_images.keys()))}

    tasks = []
    for dicom_id, selected_names in image_selected.items():
        subject_id, study_id, split = id_map[dicom_id]
        img_path = os.path.join(MIMIC_DIR, f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg")
        sg_path  = os.path.join(SG_DIR, f"{dicom_id}_SceneGraph.json")
        tasks.append((dicom_id, img_path, sg_path, split, list(selected_names), class_to_id))

    print(f"[silver] tasks: {len(tasks)}  workers: {NUM_WORKERS_SILVER}")

    records = []
    errors  = 0
    with Pool(NUM_WORKERS_SILVER) as pool:
        for i, (map_rows, err) in enumerate(
            pool.imap_unordered(process_silver_image, tasks, chunksize=32), start=1
        ):
            if err:
                print(f"  ERROR: {err}"); errors += 1
            elif map_rows:
                records.extend(map_rows)
            if i % 10000 == 0:
                print(f"  [silver {i}/{len(tasks)}] errors={errors}")

    print(f"[silver] done. records={len(records)} errors={errors}")
    return records


def convert_record(record, image_root, crop_root):
    img_path_rel   = remove_root_prefix(record["image_path"], image_root)
    crop_paths_rel = [remove_root_prefix(p, crop_root) for p in record.get("crop_image_paths", [])]
    tgt_img_path   = crop_paths_rel if len(crop_paths_rel) <= 1 else [crop_paths_rel]

    return {
        "qry_inst":     "<|image_1|> Locate the specific region that corresponds to the provided text description.",
        "qry_text":     record["class_name"],
        "qry_img_path": img_path_rel,
        "tgt_inst":     "Match the target",
        "tgt_text":     ["<|image_1|>\n"],
        "tgt_img_path": tgt_img_path,
    }


def deduplicate(data):
    seen   = set()
    result = []
    for s in data:
        key = (s.get("qry_img_path", ""), s.get("qry_text", ""))
        if key not in seen:
            seen.add(key)
            result.append(s)
    removed = len(data) - len(result)
    return result, removed


def sample_test_data(data, max_samples, seed=RANDOM_SEED):
    if len(data) <= max_samples:
        return data

    rng = random.Random(seed)

    class_to_samples = defaultdict(list)
    for s in data:
        class_to_samples[str(s.get("qry_text", "")).strip()].append(s)
    for v in class_to_samples.values():
        rng.shuffle(v)

    class_names    = list(class_to_samples.keys())
    sampled        = []
    used_paths     = set()
    used_ids       = set()

    still_has = True
    while still_has and len(sampled) < max_samples:
        still_has = False
        rng.shuffle(class_names)
        for cls in class_names:
            candidates = class_to_samples[cls]
            if not candidates:
                continue
            still_has = True
            chosen_idx = next(
                (i for i, s in enumerate(candidates)
                 if s.get("qry_img_path") not in used_paths and id(s) not in used_ids),
                None
            )
            if chosen_idx is None:
                chosen_idx = next(
                    (i for i, s in enumerate(candidates) if id(s) not in used_ids), None
                )
            if chosen_idx is None:
                continue
            chosen = candidates.pop(chosen_idx)
            sampled.append(chosen)
            used_ids.add(id(chosen))
            used_paths.add(chosen.get("qry_img_path", ""))
            if len(sampled) >= max_samples:
                break

    if len(sampled) < max_samples:
        remaining = [s for v in class_to_samples.values() for s in v if id(s) not in used_ids]
        unique_r  = [s for s in remaining if s.get("qry_img_path") not in used_paths]
        dup_r     = [s for s in remaining if s.get("qry_img_path") in used_paths]
        rng.shuffle(unique_r); rng.shuffle(dup_r)
        for s in unique_r + dup_r:
            if len(sampled) >= max_samples:
                break
            sampled.append(s)

    rng.shuffle(sampled)
    return sampled[:max_samples]


def main():
    gold_records   = run_gold()
    silver_records = run_silver()

    test_data  = [convert_record(r, TEST_IMAGE_ROOT,  TEST_CROP_ROOT)  for r in gold_records]
    train_data = [convert_record(r, TRAIN_IMAGE_ROOT, TRAIN_CROP_ROOT) for r in silver_records]

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(train_data)
    rng.shuffle(test_data)

    train_data, rm_train = deduplicate(train_data)
    test_data,  rm_test  = deduplicate(test_data)
    print(f"train dedup removed: {rm_train}  remaining: {len(train_data)}")
    print(f"test  dedup removed: {rm_test}   remaining: {len(test_data)}")

    original_test_count = len(test_data)
    test_data = sample_test_data(test_data, MAX_TEST_SAMPLES, RANDOM_SEED)
    print(f"test samples: {original_test_count} -> {len(test_data)}")

    for path, data in [(OUTPUT_TRAIN_JSON, train_data), (OUTPUT_TEST_JSON, test_data)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
