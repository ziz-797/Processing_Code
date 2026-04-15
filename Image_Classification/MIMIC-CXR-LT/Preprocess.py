
import json
import random
from pathlib import Path

import pandas as pd


# ========= Task Configurations =========
TASKS = [
    {
        "name": "test",
        "input_csv": "./mimic-cxr-lt_single-label_test.csv",
        "output_json": "",
        "sampling": True,
        "num_samples": 1000,
        "random_seed": 42,
        "tgt_text_mode": "full",   # first positive + all other class names
    },
    {
        "name": "train",
        "input_csv": "./mimic-cxr-lt_single-label_train.csv",
        "output_json": "",
        "sampling": False,
        "num_samples": None,
        "random_seed": None,
        "tgt_text_mode": "pos_only",  # only the positive class name
    },
]

# ========= Fixed text =========
QRY_INST = "<|image_1|> Represent the given image with the following question:"
QRY_TEXT = "What is the radiological abnormal findings observed in this chest X-ray?"


# ========= Helpers =========

def detect_label_columns(df: pd.DataFrame) -> list[str]:
    """
    Automatically detect label columns by excluding common metadata columns
    and keeping columns whose non-null values are all within {0, 1}.
    """
    exclude = {"subject_id", "study_id", "dicom_id", "path", "label"}
    candidate_cols = [c for c in df.columns if c not in exclude]

    label_cols = []
    for c in candidate_cols:
        s_num = pd.to_numeric(df[c], errors="coerce")
        if s_num.isna().all():
            continue
        uniq = set(s_num.dropna().unique().tolist())
        if uniq.issubset({0, 1}):
            label_cols.append(c)

    if not label_cols:
        raise ValueError(
            "Failed to detect label columns automatically. "
            "Please check whether the CSV contains 0/1 label columns."
        )
    return label_cols


def balanced_sample_by_class(
    one_pos_df: pd.DataFrame,
    label_cols: list[str],
    total_samples: int,
    random_seed: int,
):
    """
    Evenly sample total_samples across classes from one_pos_df.

    Strategy:
    1. Group rows by class.
    2. Allocate floor(total_samples / num_classes) to each class.
    3. If a class has fewer samples than its quota, take all of them.
    4. Redistribute the remaining quota to classes that still have unused samples.
    """
    rng = random.Random(random_seed)

    class_to_df = {c: one_pos_df[one_pos_df[c] == 1].copy() for c in label_cols}
    class_counts = {c: len(class_to_df[c]) for c in label_cols}
    available_total = sum(class_counts.values())

    if available_total == 0:
        raise ValueError("No available samples after filtering single-positive rows.")

    actual_total = min(total_samples, available_total)
    num_classes = len(label_cols)

    base_quota = actual_total // num_classes
    remainder = actual_total % num_classes

    ordered_classes = label_cols[:]
    target_quota = {c: base_quota for c in ordered_classes}
    for c in ordered_classes[:remainder]:
        target_quota[c] += 1

    selected_indices = []
    selected_per_class = {}
    leftover_needed = 0

    for c in ordered_classes:
        class_df = class_to_df[c]
        quota = target_quota[c]
        take_n = min(quota, len(class_df))
        selected_per_class[c] = take_n

        if take_n > 0:
            sampled_idx = class_df.sample(n=take_n, random_state=random_seed).index.tolist()
            selected_indices.extend(sampled_idx)

        if take_n < quota:
            leftover_needed += quota - take_n

    if leftover_needed > 0:
        already_selected_set = set(selected_indices)
        remaining_pool = [
            idx
            for c in ordered_classes
            for idx in class_to_df[c].index.tolist()
            if idx not in already_selected_set
        ]
        if remaining_pool:
            extra_n = min(leftover_needed, len(remaining_pool))
            selected_indices.extend(rng.sample(remaining_pool, extra_n))

    sampled_df = one_pos_df.loc[selected_indices].copy()
    sampled_df = sampled_df[~sampled_df.index.duplicated(keep="first")]

    if len(sampled_df) > actual_total:
        sampled_df = sampled_df.sample(n=actual_total, random_state=random_seed)

    sampled_df = sampled_df.reset_index(drop=True)
    sampled_class_counts = {c: int((sampled_df[c] == 1).sum()) for c in ordered_classes}

    return sampled_df, class_counts, sampled_class_counts


# ========= Per-task runner =========

def run_task(task: dict) -> None:
    name = task["name"]
    in_path = Path(task["input_csv"])
    out_path = Path(task["output_json"])
    do_sampling = task["sampling"]
    num_samples = task["num_samples"]
    random_seed = task["random_seed"]
    tgt_text_mode = task["tgt_text_mode"]

    print(f"\n{'='*60}")
    print(f"[TASK] {name}")
    print(f"{'='*60}")

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)

    if "path" not in df.columns:
        raise ValueError(f"[{name}] Column 'path' not found in CSV.")

    label_cols = detect_label_columns(df)

    for c in label_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    pos_cnt = df[label_cols].sum(axis=1)
    one_pos_df = df[pos_cnt == 1].copy()

    if len(one_pos_df) == 0:
        raise ValueError(f"[{name}] No rows found with exactly one positive label.")

    single_positive_counts = {c: int((one_pos_df[c] == 1).sum()) for c in label_cols}
    print("\n[INFO] Count of samples where each label is the only positive one:")
    for k, v in single_positive_counts.items():
        print(f"  {k}: {v}")

    # Sampling or use all
    if do_sampling:
        if random_seed is not None:
            random.seed(random_seed)
        sampled_df, original_counts, sampled_counts = balanced_sample_by_class(
            one_pos_df=one_pos_df,
            label_cols=label_cols,
            total_samples=num_samples,
            random_seed=random_seed,
        )
        print("\n[INFO] Original single-positive counts by class:")
        for k, v in original_counts.items():
            print(f"  {k}: {v}")
        print("\n[INFO] Balanced sampled counts by class:")
        for k, v in sampled_counts.items():
            print(f"  {k}: {v}")
    else:
        sampled_df = one_pos_df

    # Build records
    records = []
    for _, row in sampled_df.iterrows():
        pos_classes = [c for c in label_cols if int(row[c]) == 1]
        if len(pos_classes) != 1:
            continue

        pos_class = pos_classes[0]

        if tgt_text_mode == "full":
            tgt_text = [pos_class] + [c for c in label_cols if c != pos_class]
        else:  # "pos_only"
            tgt_text = [pos_class]

        records.append({
            "qry_inst": QRY_INST,
            "qry_text": QRY_TEXT,
            "qry_img_path": str(row["path"]),
            "tgt_text": tgt_text,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Total input rows        : {len(df)}")
    print(f"[OK] Number of label columns : {len(label_cols)}")
    print(f"[OK] Single-positive rows    : {len(one_pos_df)}")
    if do_sampling:
        print(f"[OK] Requested samples       : {num_samples}")
    print(f"[OK] Actual output records   : {len(records)}")
    print(f"[OK] Output file             : {out_path}")


# ========= Entry point =========

def main():
    for task in TASKS:
        run_task(task)

    print(f"\n{'='*60}")
    print("[DONE] All tasks completed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
