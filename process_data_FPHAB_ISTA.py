import glob
import torch
import numpy as np
from scipy.signal import medfilt
import os
from tqdm import tqdm
import random
import argparse
from collections import defaultdict
import shutil
from math import floor

# Define label_map
label_map = { 
    "wirte": 1, "wash_sponge": 2, "use_flash": 3, "use_calculator": 5, 
    "unfold_glasses": 4, "toast_wine": 6, "tear_paper": 7, 
    "take_letter_from_enveloppe": 8, "stir": 9, "squeeze_sponge": 10, 
    "squeeze_paper": 11, "sprinkle": 12, "scratch_sponge": 13, 
    "scoop_spoon": 14, "receive_coin": 15, "read_letter": 16, 
    "put_tea_bag":17, "put_sugar":18, "put_salt":19, "prick":20, 
    "pour_wine":21, "pour_milk":22, "pour_liquid_soap":23, 
    "pour_juice_bottle":24, "open_wallet":25, "open_soda_can":26, 
    "open_peanut_butter":27, "open_milk":28, "open_liquid_soap":29, 
    "open_letter":30, "open_juice_bottle":31, "light_candle":32, 
    "high_five":33, "handshake":34, "give_coin":35, "give_card":36, 
    "flip_sponge":37, "flip_pages":38, "drink_mug":39, "close_peanut_butter":40, 
    "close_milk":41, "close_liquid_soap":42, "close_juice_bottle":43, 
    "clean_glasses":44, "charge_cell_phone":45 
}

def get_label_path(path):
    parent_folder = os.path.basename(os.path.dirname(path))
    return parent_folder

def process_data(root_folder, split, frame_num=120):
    data = {'pose': [], 'coarse_label': []}
    temp_pose_list = []
    temp_label_list = []

    total_files = sum(1 for subdir, dirs, files in os.walk(root_folder) 
                     for file in files if file.endswith('.txt'))

    file_iterator = tqdm(
        [(subdir, file) for subdir, dirs, files in os.walk(root_folder) 
         for file in files if file.endswith('.txt')],
        total=total_files,
        desc=f"Processing {split} files"
    )

    for subdir, file in file_iterator:
        file_path = os.path.join(subdir, file)
        try:
            labels = get_label_path(file_path)
            mapped_label = label_map.get(labels, None)

            if mapped_label is None:
                file_iterator.set_postfix_str(f"Skipped: {file} (invalid label)")
                continue

            p = np.loadtxt(file_path).astype('float32')
            p = p.reshape(-1, 3)

            for j in range(p.shape[1]):
                p[:, j] = medfilt(p[:, j])
            p = np.expand_dims(p, axis=0)
            temp_pose_list.append(p)
            temp_label_list.append(mapped_label)

            if len(temp_pose_list) == frame_num:
                if all(label == temp_label_list[0] for label in temp_label_list):
                    data['pose'].append(temp_pose_list)
                    data['coarse_label'].append(temp_label_list[0])
                    file_iterator.set_postfix_str(f"Added batch: label {temp_label_list[0]}")
                else:
                    file_iterator.set_postfix_str("Skipped batch: mixed labels")

                temp_pose_list = []
                temp_label_list = []

        except Exception as e:
            file_iterator.set_postfix_str(f"Error: {str(e)[:30]}...")
            continue

    if len(temp_pose_list) != 0:
        print(f"Discarded {len(temp_pose_list)} remaining poses that do not form a complete set of {frame_num}")

    pose_array = np.array(data['pose'], dtype=np.float32)
    pose_tensor = torch.tensor(pose_array, dtype=torch.float32)
    label_tensor = torch.tensor(data['coarse_label'], dtype=torch.long)

    pose_tensor = pose_tensor.squeeze(3)
    assert pose_tensor.dim() == 5, f"Expected 5 dimensions but got {pose_tensor.dim()} dimensions"
    pose_tensor = pose_tensor.permute(0, 4, 1, 3, 2)
    print(f"Pose tensor shape: {pose_tensor.shape}")
    print(f"Label tensor shape: {label_tensor.shape}")
    
    return pose_tensor, label_tensor

# Function to split data
def split_data_by_label(root_dir, dest_dir, frame_num=120):
    os.makedirs(dest_dir, exist_ok=True)
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

    label_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    for label in tqdm(label_folders, desc="Splitting by label"):
        src_folder = os.path.join(root_dir, label)
        txt_files = sorted([f for f in os.listdir(src_folder) if f.endswith('.txt')])
        
        sequences = []
        for i in range(0, len(txt_files) - frame_num + 1, frame_num):
            sequences.append(txt_files[i:i+frame_num])
        
        n = len(sequences)
        train_end = floor(n * 0.7)
        val_end = floor(n * 0.85)

        split_data = {
            'train': sequences[:train_end],
            'val': sequences[train_end:val_end],
            'test': sequences[val_end:]
        }

        for split in splits:
            split_label_folder = os.path.join(dest_dir, split, label)
            os.makedirs(split_label_folder, exist_ok=True)
            for sequence in split_data[split]:
                for file in sequence:
                    src_file = os.path.join(src_folder, file)
                    dst_file = os.path.join(split_label_folder, file)
                    shutil.copyfile(src_file, dst_file)

    print("✅ Data splitting complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process FPHAB dataset")
    parser.add_argument('--root', type=str, help='Path to root folder containing pickle files.', default='/media/DATA1/NAMPV/Hand_pose_3d/FPHAB_Hamuco_pred/')
    parser.add_argument('--dest', type=str, help='Destination path to save processed files.', default='/media/DATA2/NAMPV/Hand_recognition/DD_NET/data_ista_hamuco_fphab/')
    parser.add_argument('--frames', type=int, help='Number of frames per sequence', default=120)
    
    args = parser.parse_args()

    # Split the data first
    split_data_by_label(args.root, args.dest, frame_num=args.frames)

    # Now process the data after splitting
    a, b = process_data(os.path.join(args.dest, 'val'), 'val', args.frames)
    print(a.shape)
    print(b.shape)
    torch.save(a.clone(), os.path.join(args.dest, 'val', 'data.pth'))
    torch.save(b.clone(), os.path.join(args.dest, 'val', 'gt.pth'))

    c, d = process_data(os.path.join(args.dest, 'train'), 'train', args.frames)
    print(c.shape)
    print(d.shape)
    torch.save(c.clone(), os.path.join(args.dest, 'train', 'data.pth'))
    torch.save(d.clone(), os.path.join(args.dest, 'train', 'gt.pth'))

    e, _ = process_data(os.path.join(args.dest, 'test'), 'test', args.frames)
    print(e.shape)
    torch.save(e.clone(), os.path.join(args.dest, 'test', 'data.pth'))
