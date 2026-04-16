import os
import numpy as np
from PIL import Image
 

INPUT_NPZ_PATH = r"./octmnist.npz"
OUTPUT_ROOT = r""
 
SAVE_IN_LABEL_FOLDERS = True
 
DATASET_LABEL_MAPS = {
    "pathmnist": {
        0: "adipose",
        1: "background",
        2: "debris",
        3: "lymphocytes",
        4: "mucus",
        5: "smooth muscle",
        6: "normal colon mucosa",
        7: "cancer-associated stroma",
        8: "colorectal adenocarcinoma epithelium",
    },
    "chestmnist": {
        0: "atelectasis",
        1: "cardiomegaly",
        2: "effusion",
        3: "infiltration",
        4: "mass",
        5: "nodule",
        6: "pneumonia",
        7: "pneumothorax",
        8: "consolidation",
        9: "edema",
        10: "emphysema",
        11: "fibrosis",
        12: "pleural",
        13: "hernia",
    },
    "dermamnist": {
        0: "actinic keratoses and intraepithelial carcinoma",
        1: "basal cell carcinoma",
        2: "benign keratosis-like lesions",
        3: "dermatofibroma",
        4: "melanoma",
        5: "melanocytic nevi",
        6: "vascular lesions",
    },
    "octmnist": {
        0: "choroidal neovascularization",
        1: "diabetic macular edema",
        2: "drusen",
        3: "normal",
    },
    "pneumoniamnist": {
        0: "normal",
        1: "pneumonia",
    },
    "retinamnist": {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
    },
    "breastmnist": {
        0: "malignant",
        1: "normal, benign",
    },
    "bloodmnist": {
        0: "basophil",
        1: "eosinophil",
        2: "erythroblast",
        3: "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
        4: "lymphocyte",
        5: "monocyte",
        6: "neutrophil",
        7: "platelet",
    },
    "tissuemnist": {
        0: "Collecting Duct, Connecting Tubule",
        1: "Distal Convoluted Tubule",
        2: "Glomerular endothelial cells",
        3: "Interstitial endothelial cells",
        4: "Leukocytes",
        5: "Podocytes",
        6: "Proximal Tubule Segments",
        7: "Thick Ascending Limb",
    },
    "organamnist": {
        0: "bladder",
        1: "femur-left",
        2: "femur-right",
        3: "heart",
        4: "kidney-left",
        5: "kidney-right",
        6: "liver",
        7: "lung-left",
        8: "lung-right",
        9: "pancreas",
        10: "spleen",
    },
    "organcmnist": {
        0: "bladder",
        1: "femur-left",
        2: "femur-right",
        3: "heart",
        4: "kidney-left",
        5: "kidney-right",
        6: "liver",
        7: "lung-left",
        8: "lung-right",
        9: "pancreas",
        10: "spleen",
    },
    "organsmnist": {
        0: "bladder",
        1: "femur-left",
        2: "femur-right",
        3: "heart",
        4: "kidney-left",
        5: "kidney-right",
        6: "liver",
        7: "lung-left",
        8: "lung-right",
        9: "pancreas",
        10: "spleen",
    },
    "organmnist3d": {
        0: "liver",
        1: "kidney-right",
        2: "kidney-left",
        3: "femur-right",
        4: "femur-left",
        5: "bladder",
        6: "heart",
        7: "lung-right",
        8: "lung-left",
        9: "spleen",
        10: "pancreas",
    },
    "nodulemnist3d": {
        0: "benign",
        1: "malignant",
    },
    "adrenalmnist3d": {
        0: "normal",
        1: "hyperplasia",
    },
    "fracturemnist3d": {
        0: "buckle rib fracture",
        1: "nondisplaced rib fracture",
        2: "displaced rib fracture",
    },
    "vesselmnist3d": {
        0: "vessel",
        1: "aneurysm",
    },
    "synapsemnist3d": {
        0: "inhibitory synapse",
        1: "excitatory synapse",
    },
}
 

def resolve_label_map(npz_path):
    """
    Extract the dataset name from the NPZ filename and return its label map.
    Handles suffixes like _64, _128, _224 (e.g. 'tissuemnist_224.npz' -> 'tissuemnist').
    Returns None if the dataset is not recognized.
    """
    basename = os.path.basename(npz_path)           
    stem = os.path.splitext(basename)[0]             

    for suffix in ("_28", "_64", "_128", "_224"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
 
    dataset_key = stem.lower()                    
 
    if dataset_key in DATASET_LABEL_MAPS:
        print(f"[Info] Detected dataset: '{dataset_key}' — label map loaded automatically.")
        return DATASET_LABEL_MAPS[dataset_key]
    else:
        print(f"[Warning] Dataset '{dataset_key}' not found in label map registry. "
              "Folders will be named by integer label.")
        return None
 
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
 
 
def save_one_image(img_array, save_path):
    img_array = np.asarray(img_array)
 
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
 
    if img_array.ndim == 2:
        image = Image.fromarray(img_array, mode="L")
    elif img_array.ndim == 3:
        if img_array.shape[2] == 1:
            image = Image.fromarray(img_array[:, :, 0], mode="L")
        elif img_array.shape[2] == 3:
            image = Image.fromarray(img_array, mode="RGB")
        elif img_array.shape[2] == 4:
            image = Image.fromarray(img_array, mode="RGBA")
        else:
            raise ValueError(f"Unsupported number of channels: {img_array.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")
 
    image.save(save_path)
 
 
def label_to_folder_name(label_value, label_map):
    label_value = int(label_value)
    if label_map is not None and label_value in label_map:
        return label_map[label_value]
    return str(label_value)
 
 
def process_split(images, labels, split_name, label_map):
    split_root = os.path.join(OUTPUT_ROOT, split_name)
    ensure_dir(split_root)
 
    total = len(images)
    success = 0
 
    for i in range(total):
        img = images[i]
        label = labels[i]
 
        # Handle label shape=(1,)
        if isinstance(label, np.ndarray):
            if label.size == 1:
                label = int(label.reshape(-1)[0])
            else:
                raise ValueError(f"[{split_name}] Unexpected label format at index {i}: {label}")
        else:
            label = int(label)
 
        if SAVE_IN_LABEL_FOLDERS:
            class_name = label_to_folder_name(label, label_map)
            save_dir = os.path.join(split_root, class_name)
        else:
            save_dir = split_root
 
        ensure_dir(save_dir)
 
        save_name = f"{split_name}_{i:06d}.png"
        save_path = os.path.join(save_dir, save_name)
 
        try:
            save_one_image(img, save_path)
            success += 1
        except Exception as e:
            print(f"[Skip] {split_name} image {i} could not be saved: {e}")
 
    print(f"[Done] {split_name}: saved {success} / {total} images successfully.")
 

def main():
    ensure_dir(OUTPUT_ROOT)
 
    print(f"[Info] Reading NPZ file: {INPUT_NPZ_PATH}")
    data = np.load(INPUT_NPZ_PATH)
 
    print("\n[Info] Fields found in NPZ file:")
    for k in data.files:
        arr = data[k]
        print(f"  - {k}: shape={arr.shape}, dtype={arr.dtype}")
 
    required_keys = [
        "train_images", "train_labels",
        "val_images",   "val_labels",
        "test_images",  "test_labels",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"[Error] Missing required field: '{key}'")
 
    label_map = resolve_label_map(INPUT_NPZ_PATH)
 
    train_images, train_labels = data["train_images"], data["train_labels"]
    val_images,   val_labels   = data["val_images"],   data["val_labels"]
    test_images,  test_labels  = data["test_images"],  data["test_labels"]
 
    print("\n[Info] Extracting and saving images...")
    process_split(train_images, train_labels, "train", label_map)
    process_split(val_images,   val_labels,   "val",   label_map)
    process_split(test_images,  test_labels,  "test",  label_map)
 
    print(f"\n[Info] All done. Output directory: {OUTPUT_ROOT}")
 
 
if __name__ == "__main__":
    main()
