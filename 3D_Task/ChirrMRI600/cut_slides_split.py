#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


INPUT_DIR = Path("./ChirrMRI600")
OUTPUT_BASE = Path("./ChirrMRI600")

T1_CSV = INPUT_DIR / "Cirrhosis_T1_slide_folders.csv"
T2_CSV = INPUT_DIR / "Cirrhosis_T2_slide_folders.csv"

NUM_SLICES = 64
DONE_MARKER = "_DONE"
 
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.5
MIN_FOREGROUND_RATIO = 0.01
 
SOURCE_MAP = {
    "Cirrhosis_T1_slide": [
        ("Cirrhosis_T1_3D", ""),
        ("Healthy_subjects/T1_W_Healthy", "healthy_"),
    ],
    "Cirrhosis_T2_slide": [
        ("Cirrhosis_T2_3D", ""),
        ("Healthy_subjects/T2_W_Healthy", "healthy_"),
    ],
}
 
CSV_MAP = {
    "Cirrhosis_T1_slide": "t1_csv",
    "Cirrhosis_T2_slide": "t2_csv",
}
 
 
def setup_logging():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_BASE / "slice_chirrmri600.log"
 
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
 
 
def load_split_csv(csv_path: Path) -> dict:
    split_map = {}
    if not csv_path.exists():
        logging.warning(f"CSV file not found: {csv_path}")
        return split_map
 
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder_name = row["folder_name"].strip()
            split = row["split"].strip()
            split_map[folder_name] = split
 
    logging.info(f"Loaded {len(split_map)} entries from {csv_path}")
    return split_map
 
 
def discover_nii_files(input_dir: Path):
    tasks = []
 
    for output_folder, sources in SOURCE_MAP.items():
        out_root = OUTPUT_BASE / output_folder
 
        for rel_path, prefix in sources:
            src_dir = input_dir / rel_path
 
            if not src_dir.exists():
                logging.warning(f"Source folder not found: {src_dir}")
                continue
 
            nii_files = sorted(src_dir.rglob("*.nii.gz"))
            logging.info(f"Found {len(nii_files)} nii.gz files under {src_dir}")
 
            for nii_path in nii_files:
                case_id = nii_path.name.replace(".nii.gz", "")
                out_name = f"{prefix}{case_id}"
                out_dir = out_root / out_name
 
                if (out_dir / DONE_MARKER).exists():
                    logging.info(f"Skip {output_folder}/{out_name}: already done")
                    continue
 
                tasks.append((nii_path, out_dir, out_name, output_folder))
 
    return tasks
 
 
def get_slice_axis(data: np.ndarray) -> int:
    if data.ndim != 3:
        raise ValueError(f"Only 3D volume is supported, but got shape={data.shape}")
    return 2
 
 
def normalize_volume(data: np.ndarray) -> np.ndarray:
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
 
 
def get_contiguous_core_range(valid_indices):
    if len(valid_indices) == 0:
        return None
    return valid_indices[0], valid_indices[-1]
 
 
def sample_indices_from_range(start_idx, end_idx, num_slices):
    if end_idx < start_idx:
        return []
    return np.linspace(start_idx, end_idx, num_slices, dtype=int).tolist()
 
 
def process_volume(args):
    nii_path, out_dir, case_name, output_folder = args
 
    try:
        img = nib.load(str(nii_path))
        data = np.asarray(img.dataobj)
 
        if data.ndim != 3:
            return (output_folder, case_name, False, f"Expected 3D data, got shape={data.shape}")
 
        slice_axis = get_slice_axis(data)
        n_slices = data.shape[slice_axis]
 
        if n_slices <= 0:
            return (output_folder, case_name, False, f"Invalid number of slices: {n_slices}")
 
        norm_data = normalize_volume(data)
        valid_indices = get_valid_slice_indices(data, slice_axis)
 
        if len(valid_indices) == 0:
            indices = np.linspace(0, n_slices - 1, NUM_SLICES, dtype=int).tolist()
            range_msg = f"fallback_full_range=0-{n_slices - 1}"
        else:
            start_idx, end_idx = get_contiguous_core_range(valid_indices)
            indices = sample_indices_from_range(start_idx, end_idx, NUM_SLICES)
            range_msg = f"core_range={start_idx}-{end_idx}"
 
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
            output_folder,
            case_name,
            True,
            f"OK | valid_slices={len(valid_indices)}/{n_slices} | {range_msg}",
        )
 
    except Exception as e:
        return (output_folder, case_name, False, str(e))
 
 
def reorganize_by_split(split_maps: dict):
    for output_folder, csv_key in CSV_MAP.items():
        split_map = split_maps.get(csv_key, {})
        if not split_map:
            logging.warning(f"No split info for {output_folder}, skipping reorganization")
            continue
 
        out_root = OUTPUT_BASE / output_folder
 
        if not out_root.exists():
            logging.warning(f"Output folder not found: {out_root}, skipping")
            continue
 
        subdirs = [d for d in out_root.iterdir() if d.is_dir()]
        moved = 0
        unmatched = []
 
        for subdir in subdirs:
            folder_name = subdir.name
 
            if folder_name in ("train", "valid", "test"):
                continue
 
            matched_split = split_map.get(folder_name, None)
 
            if matched_split is None:
                unmatched.append(folder_name)
                shutil.rmtree(subdir)
                continue
 
            target_dir = out_root / matched_split / folder_name
            target_dir.parent.mkdir(parents=True, exist_ok=True)
 
            if target_dir.exists():
                shutil.rmtree(target_dir)
 
            shutil.move(str(subdir), str(target_dir))
            moved += 1
 
        logging.info(
            f"Reorganized {output_folder}: moved {moved} folders, "
            f"deleted {len(unmatched)} unmatched folders"
        )
 
        if unmatched:
            logging.warning(
                f"Deleted unmatched folders in {output_folder} (not found in CSV): {unmatched}"
            )
 
 
def main():
    parser = argparse.ArgumentParser(description="Slice ChirrMRI600 NIfTI volumes to PNG")
    parser.add_argument("--workers", type=int, default=8, help="number of workers")
    parser.add_argument("--num_slices", type=int, default=64, help="number of sampled slices")
    parser.add_argument("--input_dir", type=str, default=None, help="override INPUT_DIR")
    parser.add_argument("--output_dir", type=str, default=None, help="override OUTPUT_BASE")
    parser.add_argument("--t1_csv", type=str, default=None, help="override T1 split CSV path")
    parser.add_argument("--t2_csv", type=str, default=None, help="override T2 split CSV path")
    args = parser.parse_args()
 
    global NUM_SLICES, INPUT_DIR, OUTPUT_BASE, T1_CSV, T2_CSV
    NUM_SLICES = args.num_slices
    if args.input_dir:
        INPUT_DIR = Path(args.input_dir)
    if args.output_dir:
        OUTPUT_BASE = Path(args.output_dir)
    if args.t1_csv:
        T1_CSV = Path(args.t1_csv)
    if args.t2_csv:
        T2_CSV = Path(args.t2_csv)
 
    setup_logging()
 
    logging.info(f"Input dir:  {INPUT_DIR}")
    logging.info(f"Output dir: {OUTPUT_BASE}")
    logging.info(f"T1 CSV:     {T1_CSV}")
    logging.info(f"T2 CSV:     {T2_CSV}")
    logging.info(
        f"Settings -> percentile normalize: "
        f"p{LOW_PERCENTILE}-p{HIGH_PERCENTILE}, "
        f"min_fg_ratio={MIN_FOREGROUND_RATIO}, "
        f"num_slices={NUM_SLICES}, workers={args.workers}"
    )
 
    for out_folder, sources in SOURCE_MAP.items():
        for rel_path, prefix in sources:
            prefix_info = f'prefix="{prefix}"' if prefix else "no prefix"
            logging.info(f"  {rel_path} -> {out_folder}/ ({prefix_info})")
 
    split_maps = {
        "t1_csv": load_split_csv(T1_CSV),
        "t2_csv": load_split_csv(T2_CSV),
    }
 
    tasks = discover_nii_files(INPUT_DIR)
    total = len(tasks)
 
    logging.info(f"Found {total} volumes to process.")
 
    if total == 0:
        logging.info("No new volumes to slice. Proceeding to reorganization.")
    else:
        done = 0
        failed = 0
        failed_files = []
        t0 = time.time()
 
        with Pool(processes=args.workers) as pool:
            for output_folder, case_name, success, msg in pool.imap_unordered(
                process_volume, tasks
            ):
                done += 1
                if success:
                    logging.info(f"SUCCESS {output_folder}/{case_name}: {msg}")
                else:
                    failed += 1
                    failed_files.append((output_folder, case_name, msg))
                    logging.error(f"FAILED {output_folder}/{case_name}: {msg}")
 
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    logging.info(
                        f"{done}/{total} done ({done / total * 100:.1f}%) | "
                        f"{rate:.2f} vol/s | ETA {eta / 60:.1f} min"
                    )
 
        elapsed = time.time() - t0
        logging.info(
            f"Slicing done: success={done - failed}, failed={failed}, "
            f"time={elapsed / 60:.2f} min"
        )
 
        if failed_files:
            fail_path = OUTPUT_BASE / "failed_files.txt"
            with open(fail_path, "w", encoding="utf-8") as f:
                for output_folder, case_name, msg in failed_files:
                    f.write(f"{output_folder}\t{case_name}\t{msg}\n")
            logging.info(f"Failed file list saved to: {fail_path}")
 
    logging.info("Reorganizing output folders by split...")
    reorganize_by_split(split_maps)
    logging.info("All done.")
 
 
if __name__ == "__main__":
    main()
