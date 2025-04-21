import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from manopth.manopth.manolayer import ManoLayer
from collections import defaultdict
from scipy.signal import medfilt
from tqdm import tqdm


# label_map = {
#     1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
#     11: 10,  # Pliers
#     12: 11,  # Kettle
#     13: 12,  # Knife
#     14: 13,  # Trash Can
#     17: 14,  # Lamp
#     18: 15,  # Stapler
#     20: 16   # Chair
# }
label_map = {
    2: 1,
    5: 2,
    12: 3,
    17: 4
}
def get_label_from_path(path):
    parts = path.split('/')
    for part in parts:
        if part.startswith('C') and part[1:].isdigit():
            return int(part[1:])
    return None

def save_dataset_splits(data, ground_truth, args):
    total_samples = len(data)
    train_size = int(0.69 * total_samples)
    val_size = int(0.16 * total_samples)
    test_size = total_samples - train_size - val_size
    
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = data[train_indices]
    train_gt = ground_truth[train_indices]
    val_data = data[val_indices]
    val_gt = ground_truth[val_indices]
    test_data = data[test_indices]
    
    val_data_path = os.path.join(args.dest, 'val', 'data.pth')
    val_gt_path = os.path.join(args.dest, 'val', 'gt.pth')
    train_data_path = os.path.join(args.dest, 'train', 'data.pth')
    train_gt_path = os.path.join(args.dest, 'train', 'gt.pth')
    test_data_path = os.path.join(args.dest, 'test', 'data.pth')
    
    os.makedirs(os.path.join(args.dest, 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.dest, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.dest, 'test'), exist_ok=True)
    
    torch.save(val_data.clone(), val_data_path)
    torch.save(val_gt.clone(), val_gt_path)
    torch.save(train_data.clone(), train_data_path)
    torch.save(train_gt.clone(), train_gt_path)
    torch.save(test_data.clone(), test_data_path)
    
    print(f"\nDataset splits saved:")
    print(f"Train set: {len(train_data)} samples ({len(train_data)/total_samples*100:.1f}%)")
    print(f"Val set: {len(val_data)} samples ({len(val_data)/total_samples*100:.1f}%)")
    print(f"Test set: {len(test_data)} samples ({len(test_data)/total_samples*100:.1f}%)")
    print(f"\nFiles saved at: {args.dest}")

def pad_tensor(sample_tensor, max_frame_num=120):

    if sample_tensor.size(0) < max_frame_num:
        zero_tensor = torch.zeros((max_frame_num - sample_tensor.size(0), sample_tensor.size(1), sample_tensor.size(2), sample_tensor.size(3)))
        sample_tensor = torch.cat([sample_tensor, zero_tensor], dim=0)
    

    elif sample_tensor.size(0) > max_frame_num:
        st = (sample_tensor.size(0) - max_frame_num) // 2
        sample_tensor = sample_tensor[st:st + max_frame_num, :, :, :]
    
    return sample_tensor

def process_hoi4d_dataset(root_folder):
    total_files = sum(1 for subdir, dirs, files in os.walk(root_folder) 
                      for file in files if file.endswith('.txt'))
    file_iterator = tqdm(
        [(subdir, file) for subdir, dirs, files in os.walk(root_folder) 
         for file in files if file.endswith('.txt')],
        total=total_files,
        desc="Processing files"
    )

    pose_list = []
    label_list = []
    temp_pose_list = []
    temp_label_list = []

    for subdir, file in file_iterator:
        file_path = os.path.join(subdir, file)
        try:
            label = get_label_from_path(file_path)
            mapped_label = label_map.get(label, None)
            
            if mapped_label is None:
                file_iterator.set_postfix_str(f"Skipped: {file} (invalid label)")
                continue

            p = np.loadtxt(file_path).astype('float32')
            p = p.reshape(-1, 3)  # [21, 3]
            for j in range(p.shape[1]):
                p[:, j] = medfilt(p[:, j])
            p = np.expand_dims(p, axis=0)  # [1, 21, 3]
            zero_object = np.zeros((1, 21, 3), dtype='float32')
            p = np.concatenate([p, zero_object], axis=0)  # [2, 21, 3]

            temp_pose_list.append(p)
            temp_label_list.append(mapped_label)

            if len(temp_pose_list) == 120:
                if all(label == temp_label_list[0] for label in temp_label_list):
                    batch_tensor = torch.from_numpy(np.stack(temp_pose_list))
                    batch_tensor = pad_tensor(batch_tensor)  
                    pose_list.append(batch_tensor)
                    label_list.append(temp_label_list[0])
                else:
                    file_iterator.set_postfix_str("Skipped batch: mixed labels")
                
                temp_pose_list = []
                temp_label_list = []

        except Exception as e:
            file_iterator.set_postfix_str(f"Error: {str(e)[:30]}...")
            continue


    if temp_pose_list and all(label == temp_label_list[0] for label in temp_label_list):
        batch_tensor = torch.from_numpy(np.stack(temp_pose_list))
        batch_tensor = pad_tensor(batch_tensor) 
        pose_list.append(batch_tensor)
        label_list.append(temp_label_list[0])

    if pose_list:
        data = torch.stack(pose_list, dim=0)
        ground_truth = torch.tensor(label_list)
        data = data.permute(0, 4, 1, 3, 2)
        
        print("Final data shape:", data.shape)
        print("Ground truth shape:", ground_truth.shape)
        
        return data, ground_truth

    return None, None

if __name__ == "__main__":
    root_folder = "/media/DATA2/DATA_HOI4D/handpose/handpose/"
    data, ground_truth = process_hoi4d_dataset(root_folder)
    
    if data is not None:
        class Args:
            def __init__(self):
                self.dest = "/media/DATA2/NAMPV/Hand_recognition/ISTA-Net/HOI4D_4C_ISTA/"
        
        args = Args()
        save_dataset_splits(data, ground_truth, args)
