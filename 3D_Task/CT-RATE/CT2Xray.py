#!/usr/bin/env python3

import argparse
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from PIL import Image

BASE_DIR = Path("./CT_DATA/data_volumes/dataset")
OUTPUT_BASE = Path("./CT_DATA")

SPLIT_CONFIG = {
    "train": {
        "input_dir": BASE_DIR / "train_fixed",
        "output_dir": OUTPUT_BASE / "train_fixed_drr",
    },
    "valid": {
        "input_dir": BASE_DIR / "valid_fixed",
        "output_dir": OUTPUT_BASE / "valid_fixed_drr",
    },
}

DONE_MARKER = "_DONE"

HU_MIN = -1000
HU_MAX = 1200
MU_WATER = 0.0025

BODY_THRESHOLD_HU = -700
BODY_KERNEL_SIZE = 5

RAW_PERCENTILES = (0.5, 99.5)
INVERT = True

AUTO_CROP = True
CROP_PAD = 8
CROP_THRESHOLD_PERCENTILE = 3


def hu_to_attenuation(data, hu_min=HU_MIN, hu_max=HU_MAX, mu_water=MU_WATER):
    data_clipped = np.clip(data, hu_min, hu_max)
    mu = mu_water * (data_clipped / 1000.0 + 1.0)
    return np.clip(mu, 0, None).astype(np.float32)


def create_body_mask(data, threshold_hu=BODY_THRESHOLD_HU, kernel_size=BODY_KERNEL_SIZE):
    mask = (data > threshold_hu).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for z in range(mask.shape[2]):
        sl = mask[:, :, z]
        sl = cv2.morphologyEx(sl, cv2.MORPH_CLOSE, kernel)
        sl = cv2.morphologyEx(sl, cv2.MORPH_OPEN, kernel)
        mask[:, :, z] = sl

    return mask.astype(np.float32)


def generate_drr(mu, axis, voxel_spacing, percentiles=RAW_PERCENTILES, invert=INVERT):
    spacing = float(voxel_spacing[axis])
    line_integral = np.sum(mu, axis=axis) * spacing
    projection = np.exp(-line_integral)

    p_low, p_high = np.percentile(projection, percentiles)
    projection = np.clip(projection, p_low, p_high)
    projection = (projection - p_low) / (p_high - p_low + 1e-8)

    if invert:
        projection = 1.0 - projection

    return np.clip(projection, 0, 1).astype(np.float32)


def auto_crop(img, pad=CROP_PAD, threshold_percentile=CROP_THRESHOLD_PERCENTILE):
    mask = img > np.percentile(img, threshold_percentile)
    coords = np.argwhere(mask)

    if len(coords) == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(img.shape[0], y1 + pad)
    x1 = min(img.shape[1], x1 + pad)

    return img[y0:y1, x0:x1]


def process_volume(args):
    nifti_path, out_subdir = args
    sub_name = out_subdir.name

    try:
        img = nib.load(nifti_path)
        data = np.asarray(img.dataobj).astype(np.float32)
        voxel_spacing = img.header.get_zooms()[:3]

        mu = hu_to_attenuation(data)

        body_mask = create_body_mask(data)
        mu = mu * body_mask
        del body_mask

        drr_ap = generate_drr(mu, axis=1, voxel_spacing=voxel_spacing)
        del mu, data

        drr_ap = np.flipud(drr_ap.T)

        sp_row, sp_col = float(voxel_spacing[2]), float(voxel_spacing[0])
        if abs(sp_row - sp_col) > 0.01:
            min_sp = min(sp_row, sp_col)
            scale_h = sp_row / min_sp
            scale_w = sp_col / min_sp
            new_h = int(round(drr_ap.shape[0] * scale_h))
            new_w = int(round(drr_ap.shape[1] * scale_w))
            drr_ap = cv2.resize(drr_ap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if AUTO_CROP:
            drr_ap = auto_crop(drr_ap)

        out_subdir.mkdir(parents=True, exist_ok=True)
        img_uint8 = (drr_ap * 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode="L").save(out_subdir / "drr_ap.png")

        (out_subdir / DONE_MARKER).touch()
        return (sub_name, True, "OK")

    except Exception as e:
        return (sub_name, False, str(e))


def discover_files(input_dir, output_dir):
    tasks = []
    for case_dir in sorted(input_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        for sub_dir in sorted(case_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            sub_name = sub_dir.name
            nifti_path = sub_dir / f"{sub_name}_1.nii.gz"
            if not nifti_path.exists():
                continue
            out_subdir = output_dir / sub_name
            if (out_subdir / DONE_MARKER).exists():
                continue
            tasks.append((nifti_path, out_subdir))
    return tasks


def run_split(split, workers):
    cfg = SPLIT_CONFIG[split]
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]

    logging.info(f"=== Processing {split} split (DRR) ===")
    logging.info(f"Input:  {input_dir}")
    logging.info(f"Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = discover_files(input_dir, output_dir)
    total = len(tasks)
    logging.info(f"Found {total} volumes to process (skipped already done)")

    if total == 0:
        logging.info("Nothing to do.")
        return

    done = 0
    failed = 0
    failed_files = []
    t0 = time.time()

    with Pool(processes=workers) as pool:
        for sub_name, success, msg in pool.imap_unordered(process_volume, tasks):
            done += 1
            if success:
                if done % 20 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (total - done) / rate if rate > 0 else 0
                    logging.info(
                        f"[{split}] {done}/{total} done "
                        f"({done / total * 100:.1f}%) | "
                        f"{rate:.1f} vol/s | ETA {eta / 3600:.1f}h"
                    )
            else:
                failed += 1
                failed_files.append((sub_name, msg))
                logging.error(f"[{split}] FAILED {sub_name}: {msg}")

    elapsed = time.time() - t0
    logging.info(
        f"=== {split} DRR done: {done - failed} succeeded, "
        f"{failed} failed, {elapsed / 3600:.2f}h ==="
    )

    if failed_files:
        fail_path = OUTPUT_BASE / f"{split}_drr_failed.txt"
        with open(fail_path, "w") as f:
            for name, msg in failed_files:
                f.write(f"{name}\t{msg}\n")
        logging.info(f"Failed files written to {fail_path}")


def setup_logging():
    log_path = OUTPUT_BASE / "generate_drr.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handler_file = logging.FileHandler(log_path, mode="a")
    handler_console = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[handler_file, handler_console],
    )
    sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser(description="Generate DRR X-ray from CT NIfTI volumes")
    parser.add_argument(
        "--split",
        choices=["train", "valid", "both"],
        default="both",
        help="Which split to process (default: both)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Starting DRR generation: split={args.split}, workers={args.workers}")

    if args.split == "both":
        run_split("train", args.workers)
        run_split("valid", args.workers)
    else:
        run_split(args.split, args.workers)


if __name__ == "__main__":
    main()
