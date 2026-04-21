#!/usr/bin/env python3

import argparse
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

BASE_DIR = Path("./RadGenomme-Chest-CT/dataset")
OUTPUT_BASE = Path("./RadGenomme-Chest-CT/dataset")

SPLIT_CONFIG = {
    "train": {
        "input_dir": BASE_DIR / "train_preprocessed",
        "output_dir": OUTPUT_BASE / "train_preprocessed_sliced",
    },
    "valid": {
        "input_dir": BASE_DIR / "valid_preprocessed",
        "output_dir": OUTPUT_BASE / "valid_preprocessed_sliced",
    },
}

NUM_SLICES = 64
DONE_MARKER = "_DONE"

WINDOW_LEVEL = -600
WINDOW_WIDTH = 1500
HU_MIN = WINDOW_LEVEL - WINDOW_WIDTH / 2
HU_MAX = WINDOW_LEVEL + WINDOW_WIDTH / 2


def setup_logging():
    log_path = OUTPUT_BASE / "slice_volumes.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handler_file = logging.FileHandler(log_path, mode="a")
    handler_console = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[handler_file, handler_console],
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)


def discover_files(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    tasks = []

    for nifti_path in sorted(input_dir.rglob("*_1.nii.gz")):
        if not nifti_path.is_file():
            continue

        rel_parent = nifti_path.parent.relative_to(input_dir)
        out_subdir = output_dir / rel_parent

        if (out_subdir / DONE_MARKER).exists():
            continue

        tasks.append((nifti_path, out_subdir))

    return tasks


def process_volume(args: tuple[Path, Path]) -> tuple[str, bool, str]:
    nifti_path, out_subdir = args
    case_name = str(nifti_path)

    try:
        img = nib.load(nifti_path)
        data = np.asarray(img.dataobj)

        if data.ndim < 3:
            return (case_name, False, f"Expected a 3D volume, but got shape={data.shape}")

        n_slices = data.shape[2]
        if n_slices <= 0:
            return (case_name, False, f"Invalid number of slices: {n_slices}")

        indices = np.linspace(0, n_slices - 1, NUM_SLICES, dtype=int)

        out_subdir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(indices):
            sl = data[:, :, idx].astype(np.float32)
            sl = (sl - HU_MIN) / (HU_MAX - HU_MIN) * 255.0
            sl = np.clip(sl, 0, 255).astype(np.uint8)

            img_pil = Image.fromarray(sl, mode="L")
            img_pil.save(out_subdir / f"slice_{i:03d}.png")

        (out_subdir / DONE_MARKER).touch()
        return (case_name, True, "OK")

    except Exception as e:
        return (case_name, False, str(e))


def run_split(split: str, workers: int):
    cfg = SPLIT_CONFIG[split]
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]

    logging.info(f"=== Processing split: {split} ===")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = discover_files(input_dir, output_dir)
    total = len(tasks)
    logging.info(f"Found {total} volumes to process")

    if total == 0:
        logging.info("No volumes need to be processed.")
        return

    done = 0
    failed = 0
    failed_files = []
    t0 = time.time()

    with Pool(processes=workers) as pool:
        for case_name, success, msg in pool.imap_unordered(process_volume, tasks):
            done += 1
            if success:
                if done % 20 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    logging.info(
                        f"[{split}] {done}/{total} completed "
                        f"({done / total * 100:.1f}%) | "
                        f"{rate:.2f} volumes/s | ETA {eta / 3600:.2f} h"
                    )
            else:
                failed += 1
                failed_files.append((case_name, msg))
                logging.error(f"[{split}] Failed: {case_name} | Error: {msg}")

    elapsed = time.time() - t0
    logging.info(
        f"=== Split {split} finished: {done - failed} succeeded, "
        f"{failed} failed, total time {elapsed / 3600:.2f} h ==="
    )

    if failed_files:
        fail_path = OUTPUT_BASE / f"{split}_failed_files.txt"
        with open(fail_path, "w", encoding="utf-8") as f:
            for name, msg in failed_files:
                f.write(f"{name}\t{msg}\n")
        logging.info(f"Failed file list saved to: {fail_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CT NIfTI volumes into PNG slices")
    parser.add_argument(
        "--split",
        choices=["train", "valid", "both"],
        default="both",
        help="Split to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Start processing with split={args.split}, workers={args.workers}")

    if args.split == "both":
        run_split("train", args.workers)
        run_split("valid", args.workers)
    else:
        run_split(args.split, args.workers)


if __name__ == "__main__":
    main()
