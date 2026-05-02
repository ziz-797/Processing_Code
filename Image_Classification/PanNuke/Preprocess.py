import os
import io
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image


INPUT_PARQUET_PATHS = [
    r"./data/fold1-00000-of-00001.parquet",
    r"./data/fold2-00000-of-00001.parquet",
    r"./data/fold3-00000-of-00001.parquet",
]

SPLIT_CSV_PATH = r"./PanNuke_split.csv"

OUTPUT_TRAIN_ROOT = Path(r"./train_images")
OUTPUT_TEST_ROOT = Path(r"./test_images")

IMAGE_COLUMN = "image"
CLASS_LABEL_COLUMN = "tissue"
IMAGE_EXT = ".png"

TISSUE_ID_TO_NAME: dict[int, str] = {
    0: "Adrenal Gland",
    1: "Bile Duct",
    2: "Bladder",
    3: "Breast",
    4: "Cervix",
    5: "Colon",
    6: "Esophagus",
    7: "Head and Neck",
    8: "Kidney",
    9: "Liver",
    10: "Lung",
    11: "Ovarian",
    12: "Pancreatic",
    13: "Prostate",
    14: "Skin",
    15: "Stomach",
    16: "Testis",
    17: "Thyroid",
    18: "Uterus",
}

CLASS_ORDER = list(TISSUE_ID_TO_NAME.values())


def load_split_mapping(csv_path: str) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    mapping = {}
    for _, row in df.iterrows():
        img_path = str(row["image_path"]).strip()
        split = str(row["split"]).strip().lower()
        mapping[img_path] = split
    print(f"[INFO] Loaded split CSV: {len(mapping)} entries "
          f"(train: {sum(1 for v in mapping.values() if v == 'train')}, "
          f"test: {sum(1 for v in mapping.values() if v == 'test')})")
    return mapping


def is_null_like(x):
    if x is None:
        return True
    if isinstance(x, (bytes, bytearray, dict, list, tuple, np.ndarray, Image.Image)):
        return False
    try:
        result = pd.isna(x)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return False
    except Exception:
        return False


def try_decode_pil_from_bytes(data):
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        return None


def try_decode_image(obj):
    if is_null_like(obj):
        return None
    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return try_decode_pil_from_bytes(obj)
    if isinstance(obj, np.ndarray):
        try:
            arr = obj.astype(np.uint8) if obj.dtype != np.uint8 else obj
            if arr.ndim in (2, 3):
                return Image.fromarray(arr)
        except Exception:
            return None
    if isinstance(obj, list):
        try:
            arr = np.array(obj)
            arr = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
            if arr.ndim in (2, 3):
                return Image.fromarray(arr)
        except Exception:
            return None
    if isinstance(obj, dict):
        if "bytes" in obj and obj["bytes"] is not None:
            return try_decode_pil_from_bytes(obj["bytes"])
        if "array" in obj and obj["array"] is not None:
            try:
                arr = np.array(obj["array"])
                arr = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
                if arr.ndim in (2, 3):
                    return Image.fromarray(arr)
            except Exception:
                pass
        if "path" in obj and obj["path"] and os.path.exists(str(obj["path"])):
            try:
                img = Image.open(obj["path"])
                img.load()
                return img
            except Exception:
                pass
    return None


def save_image(img: Image.Image, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if img.mode not in ["1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"]:
        img = img.convert("RGB")
    img.save(save_path)


def unique_save_path(folder: Path, stem: str, ext: str) -> Path:
    p = folder / f"{stem}{ext}"
    k = 1
    while p.exists():
        p = folder / f"{stem}_{k}{ext}"
        k += 1
    return p


def resolve_class_name(raw_label) -> str | None:
    try:
        tissue_id = int(float(str(raw_label).strip()))
    except (ValueError, TypeError):
        return None
    return TISSUE_ID_TO_NAME.get(tissue_id, None)


def process_parquet(parquet_path: Path, fold_num: int, split_mapping: dict[str, str]) -> dict:
    print(f"\n[INFO] Reading fold{fold_num}: {parquet_path.name}")
    df = pd.read_parquet(parquet_path)
    print(f"[INFO] Rows: {len(df)} | Columns: {list(df.columns)}")

    class_counts: dict[str, dict[str, int]] = {}
    saved = 0
    failed = 0
    no_split = 0

    for local_idx, row in enumerate(df.itertuples(index=False)):
        stem = f"fold{fold_num}_{local_idx:08d}"

        raw_label = getattr(row, CLASS_LABEL_COLUMN)
        if is_null_like(raw_label):
            print(f"  [WARN] {stem} has empty class label, skipped")
            failed += 1
            continue

        class_name = resolve_class_name(raw_label)
        if class_name is None:
            print(f"  [WARN] {stem} label '{raw_label}' is not in mapping table, skipped")
            failed += 1
            continue

        relative_path = f"{class_name}/{stem}{IMAGE_EXT}"
        split = split_mapping.get(relative_path, None)

        if split is None:
            alt_stem = f"fold{fold_num}_{local_idx}"
            alt_path = f"{class_name}/{alt_stem}{IMAGE_EXT}"
            split = split_mapping.get(alt_path, None)

        if split is None:
            print(f"  [WARN] {relative_path} not found in split CSV, skipped")
            no_split += 1
            failed += 1
            continue

        if split == "train":
            output_root = OUTPUT_TRAIN_ROOT
        elif split == "test":
            output_root = OUTPUT_TEST_ROOT
        else:
            print(f"  [WARN] {relative_path} has unknown split '{split}', skipped")
            failed += 1
            continue

        img = try_decode_image(getattr(row, IMAGE_COLUMN))
        if img is None:
            print(f"  [WARN] {stem} failed to decode image, skipped")
            failed += 1
            continue

        out_folder = output_root / class_name
        save_path = unique_save_path(out_folder, stem, IMAGE_EXT)

        try:
            save_image(img, save_path)
            if class_name not in class_counts:
                class_counts[class_name] = {"train": 0, "test": 0}
            class_counts[class_name][split] += 1
            saved += 1
        except Exception as e:
            print(f"  [ERROR] Failed to save ({save_path}): {e}")
            failed += 1

        done = saved + failed
        if done % 500 == 0:
            print(f"  ... Processed {done}/{len(df)} (saved {saved} / failed {failed})")

    print(f"[INFO] fold{fold_num} finished: saved {saved} / failed {failed} "
          f"(no_split_info: {no_split})")
    return class_counts


def print_summary(total_class_counts: dict):
    print("\n" + "=" * 70)
    print(f"[DONE] Train directory: {OUTPUT_TRAIN_ROOT}")
    print(f"[DONE] Test  directory: {OUTPUT_TEST_ROOT}")
    print(f"[DONE] Class statistics (total {len(total_class_counts)} classes):")
    print(f"  {'Class':<20s} {'Train':>8s} {'Test':>8s} {'Total':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    total_train = 0
    total_test = 0
    for cls in CLASS_ORDER:
        counts = total_class_counts.get(cls, {"train": 0, "test": 0})
        train_n = counts.get("train", 0)
        test_n = counts.get("test", 0)
        total_train += train_n
        total_test += test_n
        print(f"  {cls:<20s} {train_n:>8d} {test_n:>8d} {train_n + test_n:>8d}")

    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<20s} {total_train:>8d} {total_test:>8d} {total_train + total_test:>8d}")
    print("=" * 70)


def main():
    for root in [OUTPUT_TRAIN_ROOT, OUTPUT_TEST_ROOT]:
        root.mkdir(parents=True, exist_ok=True)
        for cls in CLASS_ORDER:
            (root / cls).mkdir(parents=True, exist_ok=True)

    split_mapping = load_split_mapping(SPLIT_CSV_PATH)

    total_class_counts: dict[str, dict[str, int]] = {}

    for fold_num, parquet_path_str in enumerate(INPUT_PARQUET_PATHS, start=1):
        parquet_path = Path(parquet_path_str)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        class_counts = process_parquet(parquet_path, fold_num, split_mapping)
        for cls, counts in class_counts.items():
            if cls not in total_class_counts:
                total_class_counts[cls] = {"train": 0, "test": 0}
            for split_key in ["train", "test"]:
                total_class_counts[cls][split_key] += counts.get(split_key, 0)

    print_summary(total_class_counts)


if __name__ == "__main__":
    main()
