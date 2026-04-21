import os
import numpy as np
from PIL import Image

data_root = './MRNet/MRNet-v1.0'
output_root = os.path.join(data_root, '{split}_slides')

splits = ['train', 'valid']
views = ['axial', 'coronal', 'sagittal']
num_slices_to_keep = 10

for split in splits:
    for view in views:
        input_dir = os.path.join(data_root, split, view)
        output_dir = os.path.join(data_root, f'{split}_slides', view)

        if not os.path.isdir(input_dir):
            print(f"Directory not found, skipping: {input_dir}")
            continue

        os.makedirs(output_dir, exist_ok=True)
        npy_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
        print(f"[{split}/{view}] Found {len(npy_files)} .npy files, processing...")

        for filename in npy_files:
            file_path = os.path.join(input_dir, filename)
            volume_data = np.load(file_path)

            sample_name = os.path.splitext(filename)[0]
            sample_output_dir = os.path.join(output_dir, sample_name)
            os.makedirs(sample_output_dir, exist_ok=True)

            total_slices = volume_data.shape[0]

            if total_slices <= num_slices_to_keep:
                selected_indices = np.arange(total_slices)
            else:
                selected_indices = np.linspace(0, total_slices - 1, num_slices_to_keep, dtype=int)

            for save_idx, slice_idx in enumerate(selected_indices):
                slice_2d = volume_data[slice_idx, :, :]

                if slice_2d.dtype != np.uint8:
                    slice_min = slice_2d.min()
                    slice_max = slice_2d.max()
                    if slice_max > slice_min:
                        slice_2d = (slice_2d - slice_min) / (slice_max - slice_min) * 255.0
                    else:
                        slice_2d = np.zeros_like(slice_2d)
                    slice_2d = slice_2d.astype(np.uint8)

                img = Image.fromarray(slice_2d)
                save_path = os.path.join(
                    sample_output_dir,
                    f'slice_{save_idx:02d}_orig_{slice_idx:03d}.png'
                )
                img.save(save_path)

            print(f"  Converted: {sample_name} (original {total_slices} slices, saved {len(selected_indices)})")

print("\nAll done!")
