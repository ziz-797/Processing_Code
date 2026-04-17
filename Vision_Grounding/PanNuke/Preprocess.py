import os
import io
import re
import json
import math
import random
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

PARQUET_DIR = "./PanNuke/data"
EXTRACT_IMAGE_ROOT = "./PanNuke/images"
CROP_ROOT = "./PanNuke/crops"
VIS_ROOT = "./PanNuke/visualizations"
REMOVE_PREFIX_ROOT = "./PanNuke"

REMOVE_IMAGE_ROOT_IN_JSON = False
REMOVE_CROP_ROOT_IN_JSON = False

MIN_CROP_W = 28
MIN_CROP_H = 28
MAX_VIS_SAMPLES = 50
RANDOM_SEED = 42

FILTER_HEAVILY_EXPANDED = False
HEAVY_EXPANSION_AREA_RATIO = 6.0
HEAVY_EXPANSION_W_RATIO = 2.5
HEAVY_EXPANSION_H_RATIO = 2.5

CLASS_MAP = {
    0: "Neoplastic",
    1: "Inflammatory",
    2: "Connective",
    3: "Dead",
    4: "Epithelial",
}


def normalize_path(path: str) -> str:
    return str(path).replace("\\", "/")


def remove_root_prefix(path: str, root: str) -> str:
    path = normalize_path(path)
    root = normalize_path(root).rstrip("/")
    if path == root:
        return ""
    if path.startswith(root + "/"):
        return path[len(root) + 1:]
    return path


def remove_prefix(path: str, prefix: str) -> str:
    path = normalize_path(path)
    prefix = normalize_path(prefix).rstrip("/")
    if path.startswith(prefix):
        path = path[len(prefix):]
    return path.lstrip("/")


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^\w\-\.]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def decode_image_cell(cell):
    if cell is None:
        raise ValueError("cell is None")

    data = None

    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            data = cell["bytes"]
        elif cell.get("path") is not None:
            with open(cell["path"], "rb") as f:
                data = f.read()
        else:
            raise ValueError(f"Unsupported dict cell keys: {list(cell.keys())}")
    elif isinstance(cell, (bytes, bytearray, memoryview)):
        data = bytes(cell)
    elif isinstance(cell, str):
        if os.path.isfile(cell):
            with open(cell, "rb") as f:
                data = f.read()
        else:
            raise ValueError(f"String cell is not a valid file path: {cell}")
    else:
        raise TypeError(f"Unsupported cell type: {type(cell)}")

    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def mask_to_bbox(mask_img):
    mask = np.array(mask_img.convert("L"))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def expand_bbox_to_min_size(box, img_w, img_h, min_w=28, min_h=28):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    target_w = min(max(bw, min_w), img_w)
    target_h = min(max(bh, min_h), img_h)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    nx1 = int(round(cx - target_w / 2.0))
    ny1 = int(round(cy - target_h / 2.0))
    nx2 = nx1 + target_w
    ny2 = ny1 + target_h

    if nx1 < 0:
        nx2 += -nx1
        nx1 = 0
    if ny1 < 0:
        ny2 += -ny1
        ny1 = 0
    if nx2 > img_w:
        nx1 = max(0, nx1 - (nx2 - img_w))
        nx2 = img_w
    if ny2 > img_h:
        ny1 = max(0, ny1 - (ny2 - img_h))
        ny2 = img_h

    return (max(0, int(nx1)), max(0, int(ny1)), min(img_w, int(nx2)), min(img_h, int(ny2)))


def box_size(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1), max(1, y2 - y1)


def is_heavily_expanded(min_box, final_box):
    orig_w, orig_h = box_size(min_box)
    final_w, final_h = box_size(final_box)
    orig_area = orig_w * orig_h
    final_area = final_w * final_h
    area_ratio = final_area / max(1, orig_area)
    w_ratio = final_w / max(1, orig_w)
    h_ratio = final_h / max(1, orig_h)
    heavy = (
        area_ratio >= HEAVY_EXPANSION_AREA_RATIO
        or w_ratio >= HEAVY_EXPANSION_W_RATIO
        or h_ratio >= HEAVY_EXPANSION_H_RATIO
    )
    return heavy, {
        "orig_w": orig_w, "orig_h": orig_h,
        "final_w": final_w, "final_h": final_h,
        "orig_area": orig_area, "final_area": final_area,
        "area_ratio": area_ratio, "w_ratio": w_ratio, "h_ratio": h_ratio,
    }


def draw_text_with_bg(draw, xy, text, fill=(255, 255, 255), bg=(0, 0, 0)):
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text)
        draw.rectangle(bbox, fill=bg)
    except Exception:
        draw.rectangle((x, y, x + 250, y + 16), fill=bg)
    draw.text((x, y), text, fill=fill)


def create_visualization(original_image_path, crop_path, min_box, final_box,
                          class_name, out_path, expanded, heavily_expanded=False):
    orig = Image.open(original_image_path).convert("RGB")
    crop = Image.open(crop_path).convert("RGB")
    overlay = orig.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(min_box, outline=(255, 255, 0), width=2)
    draw.rectangle(final_box, outline=(255, 0, 0), width=2)
    draw_text_with_bg(draw, (4, 4),  f"class: {class_name}")
    draw_text_with_bg(draw, (4, 22), f"min box: {min_box}")
    draw_text_with_bg(draw, (4, 40), f"final box: {final_box}")
    draw_text_with_bg(draw, (4, 58), "expanded: yes" if expanded else "expanded: no")
    draw_text_with_bg(draw, (4, 76), "heavy_expand: yes" if heavily_expanded else "heavy_expand: no")

    crop_show = crop.copy()
    max_side = max(crop_show.size)
    if max_side < 128:
        scale = max(1, math.ceil(128 / max_side))
        crop_show = crop_show.resize(
            (crop_show.width * scale, crop_show.height * scale),
            resample=Image.NEAREST
        )

    canvas_h = max(overlay.height, crop_show.height)
    canvas_w = overlay.width + 10 + crop_show.width
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(overlay, (0, 0))
    canvas.paste(crop_show, (overlay.width + 10, 0))
    safe_mkdir(os.path.dirname(out_path))
    canvas.save(out_path)


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return [x]


def get_class_name(category_id):
    try:
        category_id = int(category_id)
    except Exception:
        return f"class_{category_id}"
    return CLASS_MAP.get(category_id, f"class_{category_id}")


def make_dedup_key(sample: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    image_path = normalize_path(sample.get("image_path", ""))
    crop_image_paths = tuple(normalize_path(p) for p in sample.get("crop_image_paths", []))
    return image_path, crop_image_paths


def deduplicate_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique_samples = []
    for sample in samples:
        key = make_dedup_key(sample)
        if key in seen:
            continue
        seen.add(key)
        unique_samples.append(sample)
    return unique_samples


def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    image_path = sample.get("image_path", "")
    class_name = sample.get("class_name", "")
    crop_image_paths = sample.get("crop_image_paths", [])

    if not isinstance(crop_image_paths, list):
        raise ValueError(f"crop_image_paths must be a list, got: {type(crop_image_paths)}")

    return {
        "qry_inst": "<|image_1|> Locate the specific region that corresponds to the provided text description.",
        "qry_text": class_name,
        "qry_img_path": remove_prefix(image_path, REMOVE_PREFIX_ROOT),
        "tgt_inst": "Match the target",
        "tgt_text": ["<|image_1|>\n"],
        "tgt_img_path": [remove_prefix(p, REMOVE_PREFIX_ROOT) for p in crop_image_paths],
    }


def split_and_convert(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_data, test_data = [], []
    for sample in samples:
        converted = convert_sample(sample)
        if sample.get("is_expanded", False):
            train_data.append(converted)
        else:
            test_data.append(converted)
    return train_data, test_data


def main():
    random.seed(RANDOM_SEED)

    safe_mkdir(EXTRACT_IMAGE_ROOT)
    safe_mkdir(CROP_ROOT)
    safe_mkdir(VIS_ROOT)

    parquet_paths = sorted(Path(PARQUET_DIR).rglob("*.parquet"))
    if not parquet_paths:
        print(f"[ERROR] No parquet files found under {PARQUET_DIR}.")
        return

    all_records = []
    vis_candidates_expanded = []
    vis_candidates_normal = []

    total_images = 0
    total_masks = 0
    total_valid_masks = 0
    total_expanded_masks = 0
    total_skipped_empty_masks = 0
    total_heavily_expanded_masks = 0
    total_filtered_heavily_expanded_masks = 0

    for parquet_path in tqdm(parquet_paths, desc="Processing parquet files"):
        parquet_path = parquet_path.resolve()
        rel_no_suffix = parquet_path.relative_to(Path(PARQUET_DIR).resolve()).with_suffix("")
        parquet_tag = sanitize_name(str(rel_no_suffix))

        image_save_dir = os.path.join(EXTRACT_IMAGE_ROOT, parquet_tag)
        crop_save_dir = os.path.join(CROP_ROOT, parquet_tag)
        safe_mkdir(image_save_dir)
        safe_mkdir(crop_save_dir)

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"[WARNING] Failed to read parquet, skipping: {parquet_path}\n  Reason: {e}")
            continue

        required_cols = {"image", "instances", "categories"}
        if not required_cols.issubset(df.columns):
            print(f"[WARNING] Missing required columns, skipping: {parquet_path}")
            print(f"  Available columns: {list(df.columns)}")
            continue

        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Rows in {parquet_path.name}", leave=False):
            try:
                image_pil = decode_image_cell(row["image"]).convert("RGB")
            except Exception as e:
                print(f"[WARNING] Failed to decode image, skipping {parquet_path.name} row={row_idx}: {e}")
                continue

            total_images += 1

            image_filename = f"{parquet_tag}_{row_idx:06d}.png"
            image_abs_path = os.path.join(image_save_dir, image_filename)
            image_pil.save(image_abs_path)

            image_path_for_record = (
                remove_root_prefix(image_abs_path, EXTRACT_IMAGE_ROOT)
                if REMOVE_IMAGE_ROOT_IN_JSON else normalize_path(image_abs_path)
            )

            instances = ensure_list(row["instances"])
            categories = ensure_list(row["categories"])

            if len(instances) != len(categories):
                print(
                    f"[WARNING] instances and categories length mismatch: "
                    f"{parquet_path.name} row={row_idx}, "
                    f"instances={len(instances)}, categories={len(categories)}. "
                    f"Processing up to the shorter length."
                )

            n = min(len(instances), len(categories))
            img_w, img_h = image_pil.size
            class_to_crop_info = defaultdict(lambda: {"crop_image_paths": [], "expanded_flags": []})

            for inst_idx in range(n):
                total_masks += 1
                inst_cell = instances[inst_idx]
                cat = categories[inst_idx]
                class_name = get_class_name(cat)

                try:
                    mask_pil = decode_image_cell(inst_cell).convert("L")
                except Exception as e:
                    print(f"[WARNING] Failed to decode mask, skipping {parquet_path.name} row={row_idx} inst={inst_idx}: {e}")
                    continue

                min_box = mask_to_bbox(mask_pil)
                if min_box is None:
                    total_skipped_empty_masks += 1
                    continue

                total_valid_masks += 1

                final_box = expand_bbox_to_min_size(min_box, img_w, img_h, min_w=MIN_CROP_W, min_h=MIN_CROP_H)
                expanded = final_box != min_box
                if expanded:
                    total_expanded_masks += 1

                heavily_expanded, expand_info = is_heavily_expanded(min_box, final_box)
                if heavily_expanded:
                    total_heavily_expanded_masks += 1

                if FILTER_HEAVILY_EXPANDED and heavily_expanded:
                    total_filtered_heavily_expanded_masks += 1
                    continue

                crop = image_pil.crop(final_box)
                class_name_safe = sanitize_name(class_name.lower())
                crop_filename = f"{parquet_tag}_{row_idx:06d}_{class_name_safe}_{inst_idx:03d}.png"
                crop_abs_path = os.path.join(crop_save_dir, crop_filename)
                crop.save(crop_abs_path)

                crop_path_for_record = (
                    remove_root_prefix(crop_abs_path, CROP_ROOT)
                    if REMOVE_CROP_ROOT_IN_JSON else normalize_path(crop_abs_path)
                )

                class_to_crop_info[class_name]["crop_image_paths"].append(crop_path_for_record)
                class_to_crop_info[class_name]["expanded_flags"].append(bool(expanded))

                vis_item = {
                    "original_image_path": image_abs_path,
                    "crop_path": crop_abs_path,
                    "class_name": class_name,
                    "min_box": min_box,
                    "final_box": final_box,
                    "expanded": expanded,
                    "heavily_expanded": heavily_expanded,
                    "expand_info": expand_info,
                    "parquet_tag": parquet_tag,
                    "row_idx": int(row_idx),
                    "inst_idx": int(inst_idx),
                }
                if expanded:
                    vis_candidates_expanded.append(vis_item)
                else:
                    vis_candidates_normal.append(vis_item)

            for class_name, info in class_to_crop_info.items():
                crop_list = info["crop_image_paths"]
                expanded_flags = info["expanded_flags"]
                if not crop_list:
                    continue
                all_records.append({
                    "image_path": image_path_for_record,
                    "class_name": class_name,
                    "crop_image_paths": crop_list,
                    "is_expanded": any(expanded_flags),
                    "expanded_flags": expanded_flags,
                })

    unique_records = deduplicate_samples(all_records)
    train_data, test_data = split_and_convert(unique_records)

    rng = random.Random(RANDOM_SEED)
    selected_vis = []
    if len(vis_candidates_expanded) >= MAX_VIS_SAMPLES:
        selected_vis = rng.sample(vis_candidates_expanded, MAX_VIS_SAMPLES)
    else:
        selected_vis.extend(vis_candidates_expanded)
        remain = MAX_VIS_SAMPLES - len(selected_vis)
        if remain > 0 and vis_candidates_normal:
            selected_vis.extend(rng.sample(vis_candidates_normal, min(remain, len(vis_candidates_normal))))

    for idx, item in enumerate(selected_vis):
        if item["heavily_expanded"]:
            prefix = "heavy_expanded"
        elif item["expanded"]:
            prefix = "expanded"
        else:
            prefix = "normal"

        vis_name = (
            f"{idx:03d}_{prefix}_{sanitize_name(item['class_name'].lower())}_"
            f"{item['parquet_tag']}_{item['row_idx']:06d}_{item['inst_idx']:03d}.png"
        )
        create_visualization(
            original_image_path=item["original_image_path"],
            crop_path=item["crop_path"],
            min_box=item["min_box"],
            final_box=item["final_box"],
            class_name=item["class_name"],
            out_path=os.path.join(VIS_ROOT, vis_name),
            expanded=item["expanded"],
            heavily_expanded=item["heavily_expanded"],
        )

    print("\n========== Done ==========")
    print(f"[INFO] Parquet files processed: {len(parquet_paths)}")
    print(f"[INFO] Images saved: {total_images}")
    print(f"[INFO] Total masks: {total_masks}")
    print(f"[INFO] Valid masks: {total_valid_masks}")
    print(f"[INFO] Empty masks skipped: {total_skipped_empty_masks}")
    print(f"[INFO] Expanded masks: {total_expanded_masks}")
    print(f"[INFO] Heavily expanded boxes: {total_heavily_expanded_masks}")
    if FILTER_HEAVILY_EXPANDED:
        print(f"[INFO] Filtered heavily expanded boxes: {total_filtered_heavily_expanded_masks}")
    else:
        print(f"[INFO] Heavy expansion filtering disabled. Heavily expanded boxes counted: {total_heavily_expanded_masks}")
    print(f"[INFO] Raw records before dedup: {len(all_records)}")
    print(f"[INFO] Records after dedup: {len(unique_records)}")
    print(f"[INFO] Train samples: {len(train_data)}")
    print(f"[INFO] Test samples: {len(test_data)}")
    print(f"[INFO] Visualization samples: {len(selected_vis)}")
    print(f"[INFO] Visualization directory: {VIS_ROOT}")
    print(f"[INFO] Image directory: {EXTRACT_IMAGE_ROOT}")
    print(f"[INFO] Crop directory: {CROP_ROOT}")


if __name__ == "__main__":
    main()
