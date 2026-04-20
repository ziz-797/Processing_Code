import json
import base64
import pandas as pd
from pathlib import Path
from io import BytesIO
from collections import defaultdict


INPUT_DIR = "./ROCO-QA/data"
OUTPUT_DIR = "./ROCO-QA/example"

FILE_ORDER = [
    "Vaild-00000-of-00001-1ddc94eb806f44ae.parquet",
    "Train-00000-of-00001-91a106d1a22c7cc9.parquet",
    "Valid-00001-of-00002.parquet",
    "Test-00000-of-00001-91a106d1a22c7cc9.parquet",
    "Valid-00000-of-00002.parquet",
]
 
def extract_split_name(parquet_path) -> str:
    return Path(parquet_path).stem.split("-")[0]
 
 
def extract_image_bytes(image_field) -> bytes:
    if isinstance(image_field, dict):
        return image_field["bytes"]
    elif isinstance(image_field, bytes):
        return image_field
    elif isinstance(image_field, str):
        return base64.b64decode(image_field)
    else:
        buf = BytesIO()
        image_field.save(buf, format="JPEG")
        return buf.getvalue()
 
 
def process_parquet_files(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
 
    parquet_files = [input_dir / f for f in FILE_ORDER]
    for pq_file in parquet_files:
        if not pq_file.exists():
            print(f"File not found: {pq_file}")
            return
 
    print(f"Found {len(parquet_files)} parquet file(s)")
    for pq_file in parquet_files:
        print(f"  Order: {pq_file.name}")
 
    final_json = []
    global_seq = 0
 
    for pq_file in parquet_files:
        split_name = extract_split_name(pq_file)
        print(f"Processing: {pq_file.name}  ->  split = {split_name}")
 
        df = pd.read_parquet(pq_file)
        print(f"  Rows: {len(df)}")
 
        for _, row in df.iterrows():
            global_seq += 1
 
            image_name = f"{split_name}_sample_{global_seq:06d}.jpg"
            image_path = images_dir / image_name
 
            img_bytes = extract_image_bytes(row["image"])
            with open(image_path, "wb") as f:
                f.write(img_bytes)
 
            final_json.append({
                "Question": row["question"],
                "Answer": row["answer"],
                "ImageName": image_name,
                "split": split_name,
            })
 
    json_path = output_dir / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)
 
    print(f"Done. Total images: {len(final_json)}")
    print(f"Images saved to: {images_dir}")
    print(f"JSON saved to:   {json_path}")
 
 
if __name__ == "__main__":
    process_parquet_files(INPUT_DIR, OUTPUT_DIR)
 
