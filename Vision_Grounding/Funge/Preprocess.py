import os
import csv
import json
import shutil
import re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from typing import List, Dict, Any, Set, Tuple

DOWNLOAD_ROOT = "./Funge"
EXTRACT_ROOT = "./Funge"
OUTPUT_IMAGE_DIR = "./Funge/funge_images"
IMAGE_SIZE = 512
MIN_WIDTH = 28
MIN_HEIGHT = 28

DATASET_CONFIGS = [
    {
        "bench_class": "IdribMSBench",
        "module": "medsegbench",
        "npz_stem": "idrib_512",
        "dataset_name": "Optic_Disc",
    }
]
 
 
def to_uint8(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size > 0 and arr.min() >= 0 and arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)
 
 
def save_image(arr, save_path):
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
 
 
def is_binary_mask(arr):
    unique_vals = set(np.unique(arr).tolist())
    return unique_vals.issubset({0, 1})
 
 
def key_to_subdir(key):
    parts = key.split("_")
    if len(parts) >= 2 and parts[0] in {"train", "val", "valid", "validation", "test"}:
        split = parts[0]
        rest = "_".join(parts[1:])
        return Path(split) / rest
    return Path(key)
 
 
def mask_to_bbox(mask_arr):
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
 
 
def step1_download_datasets():
    print("\n" + "=" * 70)
    print("STEP 1: Download datasets")
    print("=" * 70)
 
    import importlib
 
    for cfg in DATASET_CONFIGS:
        mod = importlib.import_module(cfg["module"])
        bench_cls = getattr(mod, cfg["bench_class"])
        print(f"\n[Download] {cfg['bench_class']} -> {DOWNLOAD_ROOT}")
        _ = bench_cls(
            root=DOWNLOAD_ROOT,
            split="train",
            download=True,
            size=IMAGE_SIZE,
        )
        print(f"[Done] {cfg['bench_class']} download/verification complete")
 
 
def step2_extract_npz(cfg) -> Path:
    npz_path = Path(DOWNLOAD_ROOT) / f"{cfg['npz_stem']}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
 
    out_root = Path(EXTRACT_ROOT) / cfg["npz_stem"]
    out_root.mkdir(parents=True, exist_ok=True)
 
    print(f"\n[Extract] {npz_path}")
    data = np.load(str(npz_path), allow_pickle=True)
 
    print("  Keys:")
    for key in data.files:
        arr = data[key]
        print(f"    - {key}: shape={arr.shape}, dtype={arr.dtype}")
 
    item_count = 0
 
    for key in data.files:
        arr = np.asarray(data[key])
        subdir = key_to_subdir(key)
        target_dir = out_root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
 
        if arr.ndim == 3 and arr.shape[-1] not in (1, 3, 4):
            for i in range(arr.shape[0]):
                single = arr[i]
                if is_binary_mask(single):
                    np.save(target_dir / f"{i:05d}.npy", single)
                else:
                    save_image(single, target_dir / f"{i:05d}.png")
                item_count += 1
        elif arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
            for i in range(arr.shape[0]):
                save_image(arr[i], target_dir / f"{i:05d}.png")
                item_count += 1
        elif arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] in (1, 3, 4)):
            if is_binary_mask(arr):
                np.save(target_dir / "0.npy", arr)
            else:
                save_image(arr, target_dir / "0.png")
            item_count += 1
        else:
            np.save(target_dir / f"{key}.npy", arr)
            item_count += 1
 
    print(f"  Extraction complete: {item_count} items -> {out_root}")
    return out_root
 
 
def step3_crop(cfg, extract_root: Path) -> List[Tuple[str, str]]:
    print(f"\n[Crop] Dataset: {cfg['dataset_name']}")
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
 
    dataset_name = cfg["dataset_name"]
    saved_pairs = []
 
    for split_dir in sorted(extract_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
 
        if split_name.startswith("."):
            continue
 
        images_dir = split_dir / "images"
        labels_dir = None
        for label_candidate in ["labels", "label", "label_C1", "masks", "mask"]:
            candidate = split_dir / label_candidate
            if candidate.is_dir():
                labels_dir = candidate
                break
 
        if not images_dir.is_dir():
            print(f"  [Skip] No images directory: {split_dir}")
            continue
 
        if labels_dir is None:
            print(f"  [Skip] No labels directory: {split_dir}")
            continue
 
        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
        ])
 
        print(f"  [{split_name}] Image count: {len(image_files)}, labels dir: {labels_dir.name}")
 
        for img_file in image_files:
            idx_str = img_file.stem
 
            if img_file.suffix == ".npy":
                img_arr = np.load(str(img_file))
                img_arr = to_uint8(img_arr)
                if img_arr.ndim == 2:
                    pil_img = Image.fromarray(img_arr, mode="L").convert("RGB")
                elif img_arr.ndim == 3 and img_arr.shape[2] == 1:
                    pil_img = Image.fromarray(img_arr[:, :, 0], mode="L").convert("RGB")
                elif img_arr.ndim == 3:
                    pil_img = Image.fromarray(img_arr, mode="RGB")
                else:
                    continue
            else:
                try:
                    pil_img = Image.open(str(img_file)).convert("RGB")
                except Exception as e:
                    print(f"    [WARN] Failed to open image: {img_file}, {e}")
                    continue
 
            mask_arr = None
            for ext in [".npy", ".png", ".jpg", ".bmp", ".tif"]:
                mask_file = labels_dir / f"{idx_str}{ext}"
                if mask_file.exists():
                    if ext == ".npy":
                        mask_arr = np.load(str(mask_file))
                    else:
                        mask_arr = np.array(Image.open(str(mask_file)).convert("L"))
                    break
 
            if mask_arr is None:
                continue
 
            bbox = mask_to_bbox(mask_arr)
            if bbox is None:
                continue
 
            x_min, y_min, x_max, y_max = bbox
            w, h = pil_img.size
            x_min = max(0, min(x_min, w - 1))
            y_min = max(0, min(y_min, h - 1))
            x_max = max(0, min(x_max, w))
            y_max = max(0, min(y_max, h))
 
            if x_max <= x_min or y_max <= y_min:
                continue
 
            new_image_name = f"{split_name}_{idx_str}_{dataset_name}.png"
            crop_image_name = f"{split_name}_{idx_str}_{dataset_name}_crop.png"
 
            new_image_path = os.path.join(OUTPUT_IMAGE_DIR, new_image_name)
            crop_image_path = os.path.join(OUTPUT_IMAGE_DIR, crop_image_name)
 
            try:
                pil_img.save(new_image_path, format="PNG")
            except Exception as e:
                print(f"    [WARN] Failed to save original: {new_image_path}, {e}")
                continue
 
            crop = pil_img.crop((x_min, y_min, x_max, y_max))
            try:
                crop.save(crop_image_path, format="PNG")
            except Exception as e:
                print(f"    [WARN] Failed to save crop: {crop_image_path}, {e}")
                continue
 
            saved_pairs.append((new_image_path, crop_image_path))
 
    print(f"  Saved {len(saved_pairs)} image pairs")
    return saved_pairs
 
 
def step4_filter_small_images(saved_pairs: List[Tuple[str, str]]):
    print(f"\n[Filter] Checking {len(saved_pairs)} image pairs")
 
    deleted_count = 0
 
    for original_path, crop_path in saved_pairs:
        remove = False
 
        if os.path.isfile(crop_path):
            try:
                with Image.open(crop_path) as img:
                    w, h = img.size
                if w < MIN_WIDTH or h < MIN_HEIGHT:
                    remove = True
                    print(f"  [Small] {os.path.basename(crop_path)}: {w}x{h}")
            except (UnidentifiedImageError, OSError) as e:
                remove = True
                print(f"  [Broken] {crop_path}: {e}")
        else:
            remove = True
 
        if remove:
            deleted_count += 1
            if os.path.isfile(crop_path):
                os.remove(crop_path)
            if os.path.isfile(original_path):
                os.remove(original_path)
 
    kept = len(saved_pairs) - deleted_count
    print(f"  Total: {len(saved_pairs)} | Deleted: {deleted_count} | Kept: {kept}")
 
 
def main():
    step1_download_datasets()
 
    for cfg in DATASET_CONFIGS:
        print("\n" + "=" * 70)
        print(f"Processing dataset: {cfg['dataset_name']}")
        print("=" * 70)
 
        extract_root = step2_extract_npz(cfg)
        saved_pairs = step3_crop(cfg, extract_root)
        step4_filter_small_images(saved_pairs)
 
    print("\n" + "=" * 70)
    print("All processing complete")
    print("=" * 70)
    print(f"Output image directory: {OUTPUT_IMAGE_DIR}")
 
    if os.path.isdir(OUTPUT_IMAGE_DIR):
        all_images = [f for f in os.listdir(OUTPUT_IMAGE_DIR) if f.endswith(".png")]
        originals = [f for f in all_images if not f.endswith("_crop.png")]
        crops = [f for f in all_images if f.endswith("_crop.png")]
        print(f"Final image count: {len(all_images)} (originals: {len(originals)}, crops: {len(crops)})")
 
 
if __name__ == "__main__":
    main()
