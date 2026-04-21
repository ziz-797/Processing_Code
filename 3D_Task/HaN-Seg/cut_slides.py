#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import nrrd
from PIL import Image

INPUT_DIR = Path("./HaN-Seg/set_1")
CT_OUTPUT_BASE = Path("./HaN-Seg/CT_slides")
MR_OUTPUT_BASE = Path("./HaN-Seg/MRI_slides")

NUM_SLICES = 64
DONE_MARKER = "_DONE"

WINDOW_LEVEL = 50
WINDOW_WIDTH = 350
HU_MIN = WINDOW_LEVEL - WINDOW_WIDTH / 2
HU_MAX = WINDOW_LEVEL + WINDOW_WIDTH / 2

LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5
MIN_FOREGROUND_RATIO = 0.015


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

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


def extract_slice(data: np.ndarray, slice_axis: int, idx: int) -> np.ndarray:
    if slice_axis == 0:
        return data[idx, :, :]
    elif slice_axis == 1:
        return data[:, idx, :]
    else:
        return data[:, :, idx]


def normalize_ct(slice_2d: np.ndarray) -> np.ndarray:
    sl = slice_2d.astype(np.float32)
    sl = np.clip(sl, HU_MIN, HU_MAX)
    sl = (sl - HU_MIN) / (HU_MAX - HU_MIN) * 255.0
    return sl.astype(np.uint8)


def normalize_t1_mri_volume(data: np.ndarray) -> np.ndarray:
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
        sl = extract_slice(data, slice_axis, idx)
        if is_informative_slice(sl):
            valid_indices.append(idx)

    return valid_indices


def sample_indices_from_valid(valid_indices, num_slices):
    if len(valid_indices) == 0:
        return []

    if len(valid_indices) >= num_slices:
        pos = np.linspace(0, len(valid_indices) - 1, num_slices, dtype=int)
    else:
        pos = np.round(np.linspace(0, len(valid_indices) - 1, num_slices)).astype(int)

    return [valid_indices[p] for p in pos]


def discover_tasks(input_dir: Path, modality: str, output_base: Path):
    suffix = "_IMG_CT.nrrd" if modality == "CT" else "_IMG_MR_T1.nrrd"
    tasks = []

    for case_dir in sorted(input_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        case_id = case_dir.name
        file_path = case_dir / f"{case_id}{suffix}"

        if not file_path.exists():
            logging.warning(f"Skip {case_id} [{modality}]: expected file not found -> {file_path}")
            continue

        out_dir = output_base / case_id

        if (out_dir / DONE_MARKER).exists():
            logging.info(f"Skip {case_id} [{modality}]: already done")
            continue

        tasks.append((file_path, out_dir, case_id, modality))

    return tasks


def process_volume(args):
    file_path, out_dir, case_id, modality = args

    try:
        data, header = nrrd.read(str(file_path))
        data = np.asarray(data)

        if data.ndim != 3:
            return (case_id, modality, False, f"Expected 3D data, got shape={data.shape}")

        slice_axis = get_slice_axis(data)
        n_slices = data.shape[slice_axis]

        if n_slices <= 0:
            return (case_id, modality, False, f"Invalid number of slices: {n_slices}")

        if modality == "CT":
            indices = np.linspace(0, n_slices - 1, NUM_SLICES, dtype=int).tolist()
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, idx in enumerate(indices):
                sl = extract_slice(data, slice_axis, idx)
                sl = normalize_ct(sl)
                Image.fromarray(sl, mode="L").save(out_dir / f"slice_{i:03d}.png")

            (out_dir / DONE_MARKER).touch()
            return (case_id, modality, True, "OK")

        else:
            norm_data = normalize_t1_mri_volume(data)
            valid_indices = get_valid_slice_indices(data, slice_axis)

            if len(valid_indices) == 0:
                indices = np.linspace(0, n_slices - 1, NUM_SLICES, dtype=int).tolist()
            else:
                indices = sample_indices_from_valid(valid_indices, NUM_SLICES)

            out_dir.mkdir(parents=True, exist_ok=True)

            for i, idx in enumerate(indices):
                sl = extract_slice(norm_data, slice_axis, idx)
                Image.fromarray(sl, mode="L").save(out_dir / f"slice_{i:03d}.png")

            (out_dir / DONE_MARKER).touch()
            return (case_id, modality, True, f"OK | valid_slices={len(valid_indices)}/{n_slices}")

    except Exception as e:
        return (case_id, modality, False, str(e))


def run_modality(modality: str, output_base: Path, workers: int):
    output_base.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(INPUT_DIR, modality, output_base)
    total = len(tasks)

    logging.info(f"[{modality}] Found {total} volumes to process.")

    if total == 0:
        logging.info(f"[{modality}] Nothing to do.")
        return

    done = 0
    failed = 0
    failed_files = []
    t0 = time.time()

    with Pool(processes=workers) as pool:
        for case_id, mod, success, msg in pool.imap_unordered(process_volume, tasks):
            done += 1
            if success:
                logging.info(f"[{mod}] SUCCESS {case_id}: {msg}")
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    logging.info(
                        f"[{mod}] {done}/{total} done ({done/total*100:.1f}%) | "
                        f"{rate:.2f} vol/s | ETA {eta/60:.1f} min"
                    )
            else:
                failed += 1
                failed_files.append((case_id, msg))
                logging.error(f"[{mod}] FAILED {case_id}: {msg}")

    elapsed = time.time() - t0
    logging.info(
        f"[{modality}] All done: success={done - failed}, failed={failed}, time={elapsed/60:.2f} min"
    )

    if failed_files:
        fail_path = output_base / f"failed_{modality.lower()}_files.txt"
        with open(fail_path, "w", encoding="utf-8") as f:
            for case_id, msg in failed_files:
                f.write(f"{case_id}\t{msg}\n")
        logging.info(f"[{modality}] Failed file list saved to: {fail_path}")


def main():
    parser = argparse.ArgumentParser(description="Slice CT and MR T1 NRRD volumes to PNG")
    parser.add_argument("--workers", type=int, default=8, help="number of workers")
    parser.add_argument("--num_slices", type=int, default=64, help="number of sampled slices")
    parser.add_argument(
        "--modality", type=str, default="both", choices=["CT", "MR", "both"],
        help="which modality to process: CT, MR, or both (default: both)",
    )
    args = parser.parse_args()

    global NUM_SLICES
    NUM_SLICES = args.num_slices

    setup_logging(CT_OUTPUT_BASE / "slice_volumes.log")

    logging.info(f"Input dir:     {INPUT_DIR}")
    logging.info(f"CT output dir: {CT_OUTPUT_BASE}")
    logging.info(f"MR output dir: {MR_OUTPUT_BASE}")
    logging.info(
        f"Settings -> num_slices={NUM_SLICES}, workers={args.workers}, modality={args.modality}"
    )

    if args.modality in ("CT", "both"):
        run_modality("CT", CT_OUTPUT_BASE, args.workers)

    if args.modality in ("MR", "both"):
        run_modality("MR", MR_OUTPUT_BASE, args.workers)

    logging.info("All modalities finished.")


if __name__ == "__main__":
    main()
