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

BASE_DIR = Path("./CT_DATA/data_volumes/dataset")
OUTPUT_BASE = Path("./CT_DATA")

SPLIT_CONFIG = {
    "train": {
        "input_dir": BASE_DIR / "train_fixed",
        "output_dir": OUTPUT_BASE / "train_fixed_sliced",
    },
    "valid": {
        "input_dir": BASE_DIR / "valid_fixed",
        "output_dir": OUTPUT_BASE / "valid_fixed_sliced",
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
    sys.stdout.reconfigure(line_buffering=True)


def discover_files(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
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


def process_volume(args: tuple[Path, Path]) -> tuple[str, bool, str]:
    nifti_path, out_subdir = args
    sub_name = out_subdir.name

    try:
        img = nib.load(nifti_path)
        data = np.asarray(img.dataobj)

        n_slices = data.shape[2]
        indices = np.linspace(0, n_slices - 1, NUM_SLICES, dtype=int)

        out_subdir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(indices):
            sl = data[:, :, idx].astype(np.float32)
            sl = (sl - HU_MIN) / (HU_MAX - HU_MIN) * 255.0
            sl = np.clip(sl, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(sl, mode="L")
            img_pil.save(out_subdir / f"slice_{i:03d}.png")

        (out_subdir / DONE_MARKER).touch()
        return (sub_name, True, "OK")

    except Exception as e:
        return (sub_name, False, str(e))


def run_split(split: str, workers: int):
    cfg = SPLIT_CONFIG[split]
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]

    logging.info(f"=== Processing {split} split ===")
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
                        f"({done/total*100:.1f}%) | "
                        f"{rate:.1f} vol/s | ETA {eta/3600:.1f}h"
                    )
            else:
                failed += 1
                failed_files.append((sub_name, msg))
                logging.error(f"[{split}] FAILED {sub_name}: {msg}")

    elapsed = time.time() - t0
    logging.info(
        f"=== {split} done: {done - failed} succeeded, "
        f"{failed} failed, {elapsed/3600:.2f}h ==="
    )

    if failed_files:
        fail_path = OUTPUT_BASE / f"{split}_failed_files.txt"
        with open(fail_path, "w") as f:
            for name, msg in failed_files:
                f.write(f"{name}\t{msg}\n")
        logging.info(f"Failed files written to {fail_path}")


def main():
    parser = argparse.ArgumentParser(description="Slice CT NIfTI volumes to PNG")
    parser.add_argument(
        "--split",
        choices=["train", "valid", "both"],
        default="both",
        help="Which split to process (default: both)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Starting with split={args.split}, workers={args.workers}")

    if args.split == "both":
        run_split("train", args.workers)
        run_split("valid", args.workers)
    else:
        run_split(args.split, args.workers)


if __name__ == "__main__":
    main()
