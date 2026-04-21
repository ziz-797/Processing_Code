import numpy as np
import os
import pandas as pd
from PIL import Image
import tempfile
import shutil

NPZ_FILE = './organmnist3d_64.npz?download=1'
BASE_DIR = './Qrgan3DMINist'

# NPZ_FILE = './nodulemnist3d_64.npz?download=1'
# BASE_DIR = './NoduleMINist'

# NPZ_FILE = './synapsemnist3d_64.npz?download=1'
# BASE_DIR = './SynapseMNIST'

NUM_SLICES_TO_KEEP = 64 
SPLITS = ['train', 'test', 'val']
 
print("=" * 60)
print("Step 1: Extracting npz file")
print("=" * 60)
 
tmp_dir = tempfile.mkdtemp()
data = np.load(NPZ_FILE)
 
for key in data.files:
    save_name = os.path.join(tmp_dir, f"{key}.npy")
    np.save(save_name, data[key])
    print(f"  Extracted: {key}.npy")
 
print(f"All files extracted to temporary directory.\n")
 
print("=" * 60)
print("Step 2: Splitting images and slicing to PNG (train & test only)")
print("=" * 60)
 
for split in ['train', 'test']:
    images_path = os.path.join(tmp_dir, f'{split}_images.npy')
    if not os.path.exists(images_path):
        print(f"  [Skip] {split}_images.npy not found")
        continue
 
    images = np.load(images_path)
    output_root = os.path.join(BASE_DIR, f'{split}_slides')
    os.makedirs(output_root, exist_ok=True)
 
    print(f"\n  Processing {split}_images.npy, shape={images.shape}")
 
    for i in range(images.shape[0]):
        volume_data = images[i]
        sample_output_dir = os.path.join(output_root, f'sample_{i:05d}')
        os.makedirs(sample_output_dir, exist_ok=True)
 
        total_slices = volume_data.shape[0]
 
        if total_slices <= NUM_SLICES_TO_KEEP:
            selected_indices = np.arange(total_slices)
        else:
            selected_indices = np.linspace(0, total_slices - 1, NUM_SLICES_TO_KEEP, dtype=int)
 
        for save_idx, slice_idx in enumerate(selected_indices):
            slice_2d = volume_data[slice_idx, :, :]
 
            if slice_2d.dtype != np.uint8:
                slice_min = slice_2d.min()
                slice_max = slice_2d.max()
                if slice_max > slice_min:
                    slice_2d = (slice_2d - slice_min) / (slice_max - slice_min) * 255.0
                else:
                    slice_2d = np.zeros_like(slice_2d, dtype=np.float64)
                slice_2d = slice_2d.astype(np.uint8)
 
            img = Image.fromarray(slice_2d)
            save_path = os.path.join(sample_output_dir, f'slice_{save_idx:02d}.png')
            img.save(save_path)
 
        if i % 100 == 0:
            print(f"    Processed {i}/{images.shape[0]} samples...")
 
    print(f"  {split} slicing done -> {output_root}")
 
print("\n" + "=" * 60)
print("Step 3: Converting labels to CSV")
print("=" * 60)
 
for split in SPLITS:
    labels_path = os.path.join(tmp_dir, f'{split}_labels.npy')
    if not os.path.exists(labels_path):
        print(f"  [Skip] {split}_labels.npy not found")
        continue
 
    labels = np.load(labels_path)
    df = pd.DataFrame({
        'sample_index': range(len(labels)),
        'label_index': labels.flatten()
    })
 
    csv_path = os.path.join(BASE_DIR, f'{split}_labels.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(labels)} entries)")
 
shutil.rmtree(tmp_dir)
 
print("\n" + "=" * 60)
print("All done!")
print("=" * 60)
print(f"\nOutput structure:")
print(f"  {BASE_DIR}/")
print(f"  ├── train_slides/")
print(f"  │   ├── sample_00000/")
print(f"  │   │   ├── slice_00.png")
print(f"  │   │   └── ...")
print(f"  │   └── ...")
print(f"  ├── test_slides/")
print(f"  │   └── ...")
print(f"  ├── train_labels.csv")
print(f"  ├── test_labels.csv")
print(f"  └── val_labels.csv")
