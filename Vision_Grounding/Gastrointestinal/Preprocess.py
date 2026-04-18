import os
import re
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_A_INPUT_JSON = r"./Kvasir-SEG/kavsir_bboxes.json"
DATASET_A_IMAGE_DIR = r"./Kvasir-SEG/images"
DATASET_A_SPLIT_CSV = r"./split.csv"
DATASET_A_IMAGE_SUFFIX = ".jpg"

DATASET_B_DOWNLOAD_ROOT = r"./Gastronintestinal"
DATASET_B_DOWNLOAD_SIZE = 512
DATASET_B_NPZ_PATH = None
DATASET_B_NPZ_EXTRACT_ROOT = r"./Gastronintestinal"
DATASET_B_SPLITS = ["train", "test", "val"]

OUTPUT_IMAGE_DIR = r"./Gastronintestinal/images"

RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

random.seed(RANDOM_SEED)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def safe_int(x):
    return int(float(x))


def to_uint8(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size > 0 and arr.min() >= 0 and arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def save_image_from_array(arr, save_path):
    arr = to_uint8(arr)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            img = Image.fromarray(arr[:, :, 0], mode="L")
        elif arr.shape[2] == 3:
            img = Image.fromarray(arr, mode="RGB")
        elif arr.shape[2] == 4:
            img = Image.fromarray(arr, mode="RGBA")
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    img.save(save_path)


def crop_and_save(bbox_rows, output_image_dir):
    ensure_dir(output_image_dir)
    total = len(bbox_rows)
    saved_count = 0
    print(f"[CROP] Total rows: {total}")

    for idx, row in enumerate(bbox_rows):
        split = str(row["split"]).strip()
        image_name = str(row["image_name"]).strip()
        image_path = str(row["image_path"]).strip()
        x_min = safe_int(row["x_min"])
        y_min = safe_int(row["y_min"])
        x_max = safe_int(row["x_max"])
        y_max = safe_int(row["y_max"])

        if not os.path.exists(image_path):
            print(f"[WARN] Image not found, skip: {image_path}")
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open image: {image_path}, error: {e}")
            continue

        w, h = img.size
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(0, min(x_max, w))
        y_max = max(0, min(y_max, h))

        if x_max <= x_min or y_max <= y_min:
            print(f"[WARN] Invalid box, skip: {image_name}, box=({x_min},{y_min},{x_max},{y_max})")
            continue

        stem = os.path.splitext(image_name)[0]
        new_image_name = f"{split}_{stem}.png"
        crop_image_name = f"{split}_{stem}_polyp.png"
        new_image_path = os.path.join(output_image_dir, new_image_name)
        crop_image_path = os.path.join(output_image_dir, crop_image_name)

        try:
            img.save(new_image_path, format="PNG")
        except Exception as e:
            print(f"[WARN] Failed to save original: {new_image_path}, error: {e}")
            continue

        crop = img.crop((x_min, y_min, x_max, y_max))
        try:
            crop.save(crop_image_path, format="PNG")
        except Exception as e:
            print(f"[WARN] Failed to save crop: {crop_image_path}, error: {e}")
            continue

        saved_count += 1
        if (idx + 1) % 200 == 0 or (idx + 1) == total:
            print(f"[CROP] Processed {idx + 1}/{total}")

    print(f"[CROP] Saved {saved_count} samples")
    return saved_count


def load_split_csv(csv_path):
    split_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            split_name = parts[0].strip()
            image_filename = parts[1].strip()
            stem = os.path.splitext(image_filename)[0]
            split_map[stem] = split_name
    return split_map


def bbox_to_int(bbox):
    return (
        int(float(bbox["xmin"])),
        int(float(bbox["ymin"])),
        int(float(bbox["xmax"])),
        int(float(bbox["ymax"])),
    )


def is_valid_bbox(bbox):
    required_keys = {"xmin", "ymin", "xmax", "ymax"}
    if not isinstance(bbox, dict):
        return False
    return required_keys.issubset(set(bbox.keys()))


def bbox_area(bbox):
    try:
        x_min, y_min, x_max, y_max = bbox_to_int(bbox)
        w = max(0, x_max - x_min)
        h = max(0, y_max - y_min)
        return w * h
    except Exception:
        return -1


def select_single_bbox(bboxes):
    valid_bboxes = [b for b in bboxes if is_valid_bbox(b)]
    if not valid_bboxes:
        return None
    return max(valid_bboxes, key=bbox_area)


def pipeline_dataset_a():
    print("\n" + "=" * 60)
    print("Dataset A: Kvasir-SEG Pipeline")
    print("=" * 60)

    print(f"Loading JSON: {DATASET_A_INPUT_JSON}")
    with open(DATASET_A_INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading split CSV: {DATASET_A_SPLIT_CSV}")
    split_map = load_split_csv(DATASET_A_SPLIT_CSV)
    print(f"Split CSV entries: {len(split_map)}")

    samples = []
    missing_images = []
    no_split_info = []

    for image_id, info in data.items():
        image_name = f"{image_id}{DATASET_A_IMAGE_SUFFIX}"
        image_path = os.path.join(DATASET_A_IMAGE_DIR, image_name)

        if not os.path.isfile(image_path):
            missing_images.append(image_name)
            continue

        split_name = split_map.get(image_id)
        if split_name is None:
            no_split_info.append(image_id)
            continue

        bboxes = info.get("bbox", [])
        if not isinstance(bboxes, list) or len(bboxes) == 0:
            continue

        selected_bbox = select_single_bbox(bboxes)
        if selected_bbox is None:
            continue

        samples.append({
            "image_id": image_id,
            "image_name": image_name,
            "image_path": image_path,
            "split": split_name,
            "bbox": selected_bbox,
        })

    print(f"Valid images: {len(samples)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"No split info: {len(no_split_info)}")

    if len(samples) == 0:
        print("[ERROR] Dataset A: No valid samples found.")
        return 0

    bbox_rows = []
    for sample in samples:
        bb = sample["bbox"]
        x_min, y_min, x_max, y_max = bbox_to_int(bb)
        bbox_rows.append({
            "split": sample["split"],
            "image_name": sample["image_name"],
            "image_path": sample["image_path"],
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        })

    return crop_and_save(bbox_rows, OUTPUT_IMAGE_DIR)


def is_binary_mask(arr):
    unique_vals = np.unique(arr)
    return set(unique_vals.tolist()).issubset({0, 1})


def key_to_subdir(key):
    parts = key.split("_")
    if len(parts) >= 2 and parts[0] in {"train", "val", "valid", "validation", "test"}:
        split = parts[0]
        rest = "_".join(parts[1:])
        return Path(split) / rest
    return Path(key)


def extract_npz(npz_path, output_root):
    npz_name = Path(npz_path).stem
    npz_out_root = Path(output_root) / npz_name
    ensure_dir(npz_out_root)

    print(f"[NPZ] Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    print("[NPZ] Keys:")
    for key in data.files:
        arr = data[key]
        print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")

    for key in data.files:
        arr = np.asarray(data[key])
        subdir = key_to_subdir(key)
        target_dir = npz_out_root / subdir
        ensure_dir(target_dir)

        print(f"[NPZ] Exporting key: {key}")

        if is_binary_mask(arr):
            print(f"  [binary mask -> npy] unique={np.unique(arr)[:10].tolist()}")
            _save_npz_batch_as_npy(arr, target_dir, key)
            continue

        if arr.ndim == 2:
            save_image_from_array(arr, target_dir / "0.png")
            continue

        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            save_image_from_array(arr, target_dir / "0.png")
            continue

        if arr.ndim == 3:
            for i in range(arr.shape[0]):
                if is_binary_mask(arr[i]):
                    np.save(target_dir / f"{i:05d}.npy", arr[i])
                else:
                    save_image_from_array(arr[i], target_dir / f"{i:05d}.png")
            continue

        if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
            for i in range(arr.shape[0]):
                save_image_from_array(arr[i], target_dir / f"{i:05d}.png")
            continue

        np.save(target_dir / f"{key}.npy", arr)

    return str(npz_out_root)


def _save_npz_batch_as_npy(arr, target_dir, key):
    if arr.ndim == 2:
        np.save(target_dir / "0.npy", arr)
        return
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        np.save(target_dir / "0.npy", arr)
        return
    if arr.ndim >= 3:
        for i in range(arr.shape[0]):
            np.save(target_dir / f"{i:05d}.npy", arr[i])
        return
    np.save(target_dir / f"{key}.npy", arr)


def get_file_stem_to_path(folder, valid_exts=None):
    mapping = {}
    folder = Path(folder)
    if not folder.exists():
        print(f"[WARN] Folder does not exist: {folder}")
        return mapping
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if valid_exts is not None and p.suffix.lower() not in valid_exts:
            continue
        mapping[p.stem] = str(p)
    return mapping


def mask_to_bbox(mask):
    if mask.ndim != 2:
        raise ValueError(f"Mask should be 2D, got shape={mask.shape}")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_dir_to_bbox_rows(extracted_root, splits):
    all_rows = []

    for split in splits:
        image_dir = os.path.join(extracted_root, split, "images")
        label_dir = os.path.join(extracted_root, split, "label")

        image_map = get_file_stem_to_path(image_dir, valid_exts=IMAGE_EXTS)
        mask_map = get_file_stem_to_path(label_dir, valid_exts={".npy"})

        print(f"[MASK->BBOX] Split: {split}")
        print(f"  Image dir: {image_dir} ({len(image_map)} files)")
        print(f"  Label dir: {label_dir} ({len(mask_map)} files)")

        common_names = sorted(set(image_map.keys()) & set(mask_map.keys()))
        print(f"  Matched pairs: {len(common_names)}")

        skipped = 0
        for name in common_names:
            image_path = image_map[name]
            mask_path = mask_map[name]

            try:
                mask = np.load(mask_path)
                if mask.ndim == 3 and mask.shape[-1] == 1:
                    mask = np.squeeze(mask, axis=-1)

                bbox = mask_to_bbox(mask)
                if bbox is None:
                    skipped += 1
                    continue

                x_min, y_min, x_max, y_max = bbox
                all_rows.append({
                    "split": split,
                    "image_name": os.path.basename(image_path),
                    "image_path": image_path,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                })
            except Exception as e:
                skipped += 1
                print(f"  [ERROR] {name}: {e}")

        print(f"  Valid: {len(all_rows)}, Skipped: {skipped}")

    return all_rows


def find_npz_file(download_root):
    for root, dirs, files in os.walk(download_root):
        for f in files:
            if f.endswith(".npz"):
                return os.path.join(root, f)
    return None


def pipeline_dataset_b():
    print("\n" + "=" * 60)
    print("Dataset B: PolypGenMSBench Pipeline")
    print("=" * 60)

    print("[DOWNLOAD] Downloading PolypGenMSBench...")
    try:
        from medsegbench import PolypGenMSBench

        for split_name in ["train", "test"]:
            print(f"  Downloading split: {split_name}")
            _ = PolypGenMSBench(
                root=DATASET_B_DOWNLOAD_ROOT,
                split=split_name,
                download=True,
                size=DATASET_B_DOWNLOAD_SIZE,
            )
        print("[DOWNLOAD] Done.")
    except ImportError:
        print("[WARN] medsegbench not installed. Skipping download.")
        print("       Please install: pip install medsegbench")
    except Exception as e:
        print(f"[WARN] Download error: {e}")

    npz_path = DATASET_B_NPZ_PATH
    if npz_path is None:
        print("[NPZ] Searching for NPZ file...")
        npz_path = find_npz_file(DATASET_B_DOWNLOAD_ROOT)
        if npz_path is None:
            print("[ERROR] No NPZ file found. Dataset B pipeline aborted.")
            return 0
        print(f"[NPZ] Found: {npz_path}")

    extracted_root = extract_npz(npz_path, DATASET_B_NPZ_EXTRACT_ROOT)

    bbox_rows = mask_dir_to_bbox_rows(extracted_root, DATASET_B_SPLITS)

    if len(bbox_rows) == 0:
        print("[ERROR] Dataset B: No valid bbox rows generated.")
        return 0

    return crop_and_save(bbox_rows, OUTPUT_IMAGE_DIR)


def main():
    print("=" * 60)
    print("Integrated Pipeline: Multi-Dataset Processing")
    print("=" * 60)
    print(f"Output image dir: {OUTPUT_IMAGE_DIR}")

    ensure_dir(OUTPUT_IMAGE_DIR)

    count_a = pipeline_dataset_a()
    count_b = pipeline_dataset_b()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Dataset A (Kvasir-SEG): {count_a} samples")
    print(f"Dataset B (PolypGenMSBench): {count_b} samples")
    print(f"Total: {count_a + count_b} samples")
    print(f"All images saved to: {OUTPUT_IMAGE_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
