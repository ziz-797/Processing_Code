#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

INPUT_ROOT = Path("./BraTS2023-PED")
OUTPUT_BASE = Path("./BraTS2023-PED/slides")

DATA_SUBDIRS = [
    "ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData",
    "ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData",
]

TARGET_MODALITIES = {
    "t1n": "t1n",
    "t2w": "t2w",
}

NUM_SLICES = 64
DONE_MARKER = "_DONE"

LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5
MIN_FOREGROUND_RATIO = 0.015


def setup_logging():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_BASE / "slice_brats_ped_t1n_t2w.log"

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handler_file = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler_console = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[handler_file, handler_console],
    )

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def get_slice_axis(data: np.ndarray) -> int:
    if data.ndim != 3:
        raise ValueError(f"Only 3D volume is supported, but got shape={data.shape}")
    return 2


def normalize_mri_volume(data: np.ndarray) -> np.ndarray:
    vol = data.astype(np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    foreground = vol[vol > 0]
    if foreground.size == 0:
        return np.zeros_like(vol, dtype=np.uint8)

    lo = np.percentile(foreground, LOW_PERCENTILE)
    hi = np.percentile(foreground, HIGH_PERCENTILE)

    if hi <= lo:
        lo = float(foreground.min())
        hi = float(foreground.max())
        if hi <= lo:
            return np.zeros_like(vol, dtype=np.uint8)

    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo) * 255.0
    vol = np.clip(vol, 0, 255).astype(np.uint8)
    return vol


def is_informative_slice(slice_2d: np.ndarray) -> bool:
    sl = np.nan_to_num(slice_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    nonzero_ratio = np.count_nonzero(sl > 0) / sl.size
    return nonzero_ratio >= MIN_FOREGROUND_RATIO


def get_valid_slice_indices(data: np.ndarray, slice_axis: int):
    valid_indices = []
    n_slices = data.shape[slice_axis]

    for idx in range(n_slices):
        if slice_axis == 0:
            sl = data[idx, :, :]
        elif slice_axis == 1:
            sl = data[:, idx, :]
        else:
            sl = data[:, :, idx]

        if is_informative_slice(sl):
            valid_indices.append(idx)

    return valid_indices


def sample_indices_from_valid(valid_indices, num_slices):
    if len(valid_indices) == 0:
        return []

    if len(valid_indices) >= num_slices:
        pos = np.linspace(0, len(valid_indices) - 1, num_slices, dtype=int)
        return [valid_indices[p] for p in pos]
    else:
        pos = np.linspace(0, len(valid_indices) - 1, num_slices)
        pos = np.round(pos).astype(int)
        return [valid_indices[p] for p in pos]


def discover_brats_files():
    tasks = []

    for subdir_name in DATA_SUBDIRS:
        subdir = INPUT_ROOT / subdir_name
        if not subdir.exists():
            logging.warning(f"Skip missing directory: {subdir}")
            continue

        split_name = "train" if "TrainingData" in subdir_name else "val"

        for case_dir in sorted(subdir.iterdir()):
            if not case_dir.is_dir():
                continue

            case_id = case_dir.name

            modality_paths = {}
            for modality in TARGET_MODALITIES:
                matched = sorted(case_dir.glob(f"*-{modality}.nii.gz"))
                if len(matched) == 0:
                    logging.warning(f"Skip {case_id}: missing {modality} in {case_dir}")
                    continue
                modality_paths[modality] = matched[0]

            for modality, nii_path in modality_paths.items():
                out_dir = OUTPUT_BASE / split_name / modality / case_id
                if (out_dir / DONE_MARKER).exists():
                    logging.info(f"Skip {case_id} [{modality}]: already done")
                    continue

                tasks.append((nii_path, out_dir, case_id, split_name, modality))

    return tasks


def process_one_volume(args):
    nii_path, out_dir, case_id, split_name, modality = args

    try:
        img = nib.load(str(nii_path))
        data = np.asarray(img.dataobj)

        if data.ndim != 3:
            return (case_id, modality, False, f"Expected 3D data, got shape={data.shape}")

        slice_axis = get_slice_axis(data)
        n_slices = data.shape[slice_axis]

        if n_slices <= 0:
            return (case_id, modality, False, f"Invalid number of slices: {n_slices}")

        norm_data = normalize_mri_volume(data)
        valid_indices = get_valid_slice_indices(data, slice_axis)

        if len(valid_indices) == 0:
            indices = np.linspace(0, n_slices - 1, NUM_SLICES, dtype=int).tolist()
        else:
            indices = sample_indices_from_valid(valid_indices, NUM_SLICES)

        out_dir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(indices):
            if slice_axis == 0:
                sl = norm_data[idx, :, :]
            elif slice_axis == 1:
                sl = norm_data[:, idx, :]
            else:
                sl = norm_data[:, :, idx]

            Image.fromarray(sl, mode="L").save(out_dir / f"slice_{i:03d}.png")

        (out_dir / DONE_MARKER).touch()

        return (
            case_id,
            modality,
            True,
            f"OK | split={split_name} | valid_slices={len(valid_indices)}/{n_slices}"
        )

    except Exception as e:
        return (case_id, modality, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Slice BraTS2023-PED T1n/T2w NIfTI volumes to PNG"
    )
    parser.add_argument("--workers", type=int, default=8, help="number of workers")
    parser.add_argument("--num_slices", type=int, default=64, help="number of sampled slices")
    args = parser.parse_args()

    global NUM_SLICES
    NUM_SLICES = args.num_slices

    setup_logging()

    logging.info(f"Input root:  {INPUT_ROOT}")
    logging.info(f"Output dir:  {OUTPUT_BASE}")
    logging.info(f"Target modalities: {list(TARGET_MODALITIES.keys())}")
    logging.info(
        f"Settings -> percentile normalize: "
        f"p{LOW_PERCENTILE}-p{HIGH_PERCENTILE}, "
        f"num_slices={NUM_SLICES}, workers={args.workers}"
    )

    tasks = discover_brats_files()
    total = len(tasks)

    logging.info(f"Found {total} volumes to process.")

    if total == 0:
        logging.info("Nothing to do.")
        return

    done = 0
    failed = 0
    failed_files = []
    t0 = time.time()

    with Pool(processes=args.workers) as pool:
        for case_id, modality, success, msg in pool.imap_unordered(process_one_volume, tasks):
            done += 1
            if success:
                logging.info(f"SUCCESS {case_id} [{modality}]: {msg}")
            else:
                failed += 1
                failed_files.append((case_id, modality, msg))
                logging.error(f"FAILED {case_id} [{modality}]: {msg}")

            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logging.info(
                    f"{done}/{total} done ({done/total*100:.1f}%) | "
                    f"{rate:.2f} vol/s | ETA {eta/60:.1f} min"
                )

    elapsed = time.time() - t0
    logging.info(
        f"All done: success={done - failed}, failed={failed}, time={elapsed/60:.2f} min"
    )

    if failed_files:
        fail_path = OUTPUT_BASE / "failed_brats_ped_t1n_t2w.txt"
        with open(fail_path, "w", encoding="utf-8") as f:
            for case_id, modality, msg in failed_files:
                f.write(f"{case_id}\t{modality}\t{msg}\n")
        logging.info(f"Failed file list saved to: {fail_path}")


if __name__ == "__main__":
    main()
