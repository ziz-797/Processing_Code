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

OUTPUT_BASE = Path("./SynthRAD2023/slides")

TASKS_CONFIG = {
    "brain_ct": {
        "input_dir": Path("./SynthRAD2023/Task1/brain"),
        "modality_file": "ct.nii.gz",
        "out_subdir": "images/ct",
        "normalize": "ct_brain",
        "window_level": 40,
        "window_width": 80,
    },
    "pelvis_ct": {
        "input_dir": Path("./SynthRAD2023/Task1/pelvis"),
        "modality_file": "ct.nii.gz",
        "out_subdir": "images/ct",
        "normalize": "ct_pelvis",
        "window_level": 35,
        "window_width": 350,
    },
    "brain_mri": {
        "input_dir": Path("./SynthRAD2023/Task1/brain"),
        "modality_file": "mr.nii.gz",
        "out_subdir": "images/mri",
        "normalize": "mri",
        "low_percentile": 0.5,
        "high_percentile": 99.5,
        "min_foreground_ratio": 0.015,
    },
    "pelvis_mri": {
        "input_dir": Path("./SynthRAD2023/Task1/pelvis"),
        "modality_file": "mr.nii.gz",
        "out_subdir": "images/mri",
        "normalize": "mri",
        "low_percentile": 1.0,
        "high_percentile": 99.5,
        "min_foreground_ratio": 0.02,
    },
}

NUM_SLICES = 64
DONE_MARKER = "_DONE"


def setup_logging():
    log_path = OUTPUT_BASE / "slice_all_volumes.log"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

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


def normalize_ct(slice_2d, window_level, window_width):
    hu_min = window_level - window_width / 2
    hu_max = window_level + window_width / 2
    sl = slice_2d.astype(np.float32)
    sl = np.clip(sl, hu_min, hu_max)
    sl = (sl - hu_min) / (hu_max - hu_min) * 255.0
    return sl.astype(np.uint8)


def normalize_mri_volume(data, low_percentile, high_percentile):
    vol = data.astype(np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    foreground = vol[vol > 0]
    if foreground.size == 0:
        return np.zeros_like(vol, dtype=np.uint8)

    lo = np.percentile(foreground, low_percentile)
    hi = np.percentile(foreground, high_percentile)

    if hi <= lo:
        lo = float(foreground.min())
        hi = float(foreground.max())
        if hi <= lo:
            return np.zeros_like(vol, dtype=np.uint8)

    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo) * 255.0
    vol = np.clip(vol, 0, 255).astype(np.uint8)
    return vol


def is_informative_slice(slice_2d, min_foreground_ratio):
    sl = np.nan_to_num(slice_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    nonzero_ratio = np.count_nonzero(sl > 0) / sl.size
    return nonzero_ratio >= min_foreground_ratio


def get_valid_slice_indices(data, slice_axis, min_foreground_ratio):
    valid_indices = []
    n_slices = data.shape[slice_axis]

    for idx in range(n_slices):
        if slice_axis == 0:
            sl = data[idx, :, :]
        elif slice_axis == 1:
            sl = data[:, idx, :]
        else:
            sl = data[:, :, idx]

        if is_informative_slice(sl, min_foreground_ratio):
            valid_indices.append(idx)

    return valid_indices


def sample_indices_from_valid(valid_indices, num_slices):
    if len(valid_indices) == 0:
        return []
    pos = np.linspace(0, len(valid_indices) - 1, num_slices, dtype=int)
    return [valid_indices[p] for p in pos]


def discover_files(config, num_slices):
    tasks = []
    input_dir = config["input_dir"]
    modality_file = config["modality_file"]
    out_subdir = config["out_subdir"]

    for case_dir in sorted(input_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        case_id = case_dir.name
        file_path = case_dir / modality_file

        if not file_path.exists():
            logging.warning(f"Skip {case_id}: {modality_file} not found in {input_dir.name}")
            continue

        out_dir = OUTPUT_BASE / out_subdir / case_id
        if (out_dir / DONE_MARKER).exists():
            logging.info(f"Skip {case_id} ({out_subdir}): already done")
            continue

        tasks.append((file_path, out_dir, case_id, config, num_slices))

    return tasks


def process_volume(args):
    file_path, out_dir, case_id, config, num_slices = args

    try:
        img = nib.load(str(file_path))
        data = np.asarray(img.dataobj)

        if data.ndim != 3:
            return (case_id, False, f"Expected 3D data, got shape={data.shape}")

        slice_axis = 2
        n_slices = data.shape[slice_axis]

        if n_slices <= 0:
            return (case_id, False, f"Invalid number of slices: {n_slices}")

        norm_type = config["normalize"]

        if norm_type == "mri":
            norm_data = normalize_mri_volume(
                data, config["low_percentile"], config["high_percentile"]
            )
            valid_indices = get_valid_slice_indices(
                data, slice_axis, config["min_foreground_ratio"]
            )

            if len(valid_indices) == 0:
                indices = np.linspace(0, n_slices - 1, num_slices, dtype=int).tolist()
            else:
                indices = sample_indices_from_valid(valid_indices, num_slices)

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
            return (case_id, True, f"OK | valid_slices={len(valid_indices)}/{n_slices}")

        else:
            indices = np.linspace(0, n_slices - 1, num_slices, dtype=int)
            out_dir.mkdir(parents=True, exist_ok=True)

            wl = config["window_level"]
            ww = config["window_width"]

            for i, idx in enumerate(indices):
                if slice_axis == 0:
                    sl = data[idx, :, :]
                elif slice_axis == 1:
                    sl = data[:, idx, :]
                else:
                    sl = data[:, :, idx]

                sl = normalize_ct(sl, wl, ww)
                Image.fromarray(sl, mode="L").save(out_dir / f"slice_{i:03d}.png")

            (out_dir / DONE_MARKER).touch()
            return (case_id, True, "OK")

    except Exception as e:
        return (case_id, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Slice NIfTI volumes (CT+MRI, brain+pelvis) to PNG")
    parser.add_argument("--workers", type=int, default=8, help="number of workers")
    parser.add_argument("--num_slices", type=int, default=64, help="number of sampled slices per volume")
    args = parser.parse_args()

    num_slices = args.num_slices

    setup_logging()
    logging.info(f"Output base: {OUTPUT_BASE}")
    logging.info(f"Settings -> num_slices={num_slices}, workers={args.workers}")

    all_tasks = []
    for task_name, config in TASKS_CONFIG.items():
        tasks = discover_files(config, num_slices)
        logging.info(f"[{task_name}] found {len(tasks)} volumes to process")
        all_tasks.extend(tasks)

    total = len(all_tasks)
    logging.info(f"Total volumes to process: {total}")

    if total == 0:
        logging.info("Nothing to do.")
        return

    done = 0
    failed = 0
    failed_files = []
    t0 = time.time()

    with Pool(processes=args.workers) as pool:
        for case_id, success, msg in pool.imap_unordered(process_volume, all_tasks):
            done += 1
            if success:
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    logging.info(
                        f"{done}/{total} done ({done/total*100:.1f}%) | "
                        f"{rate:.2f} vol/s | ETA {eta/60:.1f} min"
                    )
            else:
                failed += 1
                failed_files.append((case_id, msg))
                logging.error(f"FAILED {case_id}: {msg}")

    elapsed = time.time() - t0
    logging.info(
        f"All done: success={done - failed}, failed={failed}, time={elapsed/60:.2f} min"
    )

    if failed_files:
        fail_path = OUTPUT_BASE / "failed_files.txt"
        with open(fail_path, "w", encoding="utf-8") as f:
            for case_id, msg in failed_files:
                f.write(f"{case_id}\t{msg}\n")
        logging.info(f"Failed file list saved to: {fail_path}")


if __name__ == "__main__":
    main()
