import os
import csv
import json
import random
import re

INPUT_CSV_FILES = [
    "./PublicHealthQA/data/arabic.csv",
    "./PublicHealthQA/data/chinese.csv",
    "./PublicHealthQA/data/english.csv",
    "./PublicHealthQA/data/french.csv",
    "./PublicHealthQA/data/korean.csv",
    "./PublicHealthQA/data/russian.csv",
    "./PublicHealthQA/data/spanish.csv",
    "./PublicHealthQA/data/vietnamese.csv"
]

MERGED_JSON_PATH = "./PublicHealthQA_test.json"
OUTPUT_DIR = "./output_jsons"
NEGATIVE_NUM = 29
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

URL_PATTERN = re.compile(
    r"""(?ix)
    \b(
        https?://[^\s]+
        |
        www\.[^\s]+
    )
    """
)

INVISIBLE_CHAR_PATTERN = re.compile(
    r"[\u200B\u200C\u200D\u200E\u200F\uFEFF\u202A-\u202E\u2066-\u2069]"
)

MULTI_SPACE_PATTERN = re.compile(r"\s+")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = URL_PATTERN.sub(" ", text)
    text = INVISIBLE_CHAR_PATTERN.sub("", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    return text


def read_csv_file(csv_path):
    samples = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Failed to read header: {csv_path}")

        fieldnames = [x.strip() for x in reader.fieldnames]

        if "question" not in fieldnames or "answer" not in fieldnames:
            raise ValueError(
                f"Missing required columns question/answer in file: {csv_path}\n"
                f"Current columns: {fieldnames}"
            )

        for row in reader:
            raw_question = row.get("question", "")
            raw_answer = row.get("answer", "")

            question = clean_text(raw_question)
            answer = clean_text(raw_answer)

            if not question or not answer:
                continue

            samples.append({
                "question": question,
                "answer": answer
            })

    return samples


def build_unique_answer_pool(samples):
    unique_answers = []
    seen = set()

    for item in samples:
        ans = item["answer"]
        if ans not in seen:
            seen.add(ans)
            unique_answers.append(ans)

    return unique_answers


def sample_negatives(correct_answer, unique_answers, k=29):
    candidates = [a for a in unique_answers if a != correct_answer]

    if len(candidates) < k:
        return None

    return random.sample(candidates, k)


def convert_one_csv(csv_path, output_dir):
    samples = read_csv_file(csv_path)

    if not samples:
        print(f"[WARNING] No valid samples found: {csv_path}")
        return []

    unique_answers = build_unique_answer_pool(samples)

    output_data = []
    skipped = 0

    for item in samples:
        question = item["question"]
        answer = item["answer"]

        negatives = sample_negatives(answer, unique_answers, NEGATIVE_NUM)
        if negatives is None:
            skipped += 1
            continue

        output_data.append({
            "qry_inst": "Find the answer of the given question:",
            "qry_text": question,
            "tgt_text": [answer] + negatives
        })

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Input file: {csv_path}")
    print(f"[INFO] Valid samples after cleaning: {len(samples)}")
    print(f"[INFO] Number of unique answers: {len(unique_answers)}")
    print(f"[INFO] Skipped samples due to insufficient negative answers: {skipped}")
    print(f"[INFO] Output sample count: {len(output_data)}")
    print(f"[INFO] Saved single-file output to: {output_path}")
    print("-" * 80)

    return output_data


def main():
    ensure_dir(OUTPUT_DIR)

    merged_data = []

    for csv_path in INPUT_CSV_FILES:
        if not os.path.isfile(csv_path):
            print(f"[WARNING] File does not exist, skipped: {csv_path}")
            continue

        try:
            one_json_data = convert_one_csv(csv_path, OUTPUT_DIR)
            merged_data.extend(one_json_data)
        except Exception as e:
            print(f"[ERROR] Failed to process file: {csv_path}")
            print(f"        Reason: {e}")

    with open(MERGED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"[INFO] Total merged sample count: {len(merged_data)}")
    print(f"[INFO] Saved merged JSON to: {MERGED_JSON_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
