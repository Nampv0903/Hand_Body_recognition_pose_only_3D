# N T M V C
# pkl
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from manopth.manopth.manolayer import ManoLayer
import torch
from collections import defaultdict
from scipy.signal import medfilt
import os
from tqdm import tqdm

# Define label_map
label_map = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    11: 9,  # Pliers
    12: 10,  # Kettle
    13: 11,  # Knife
    14: 12,  # Trash Can
    17: 13,  # Lamp
    18: 14,  # Stapler
    20: 15   # Chair
}

label_name = {
    0: "Toy Car",
    1: "Mug",
    2: "Laptop",
    3: "Storage Furniture",
    4: "Bottle",
    5: "Safe",
    6: "Bowl",
    7: "Bucket",
    8: "Scissors",
    9: "Pliers",
    10: "Kettle",
    11: "Knife",
    12: "Trash Can",
    13: "Lamp",
    14: "Stapler",
    15: "Chair"
}

def get_label_from_path(path):
    parts = path.split('/')
    for part in parts:
        if part.startswith('C') and part[1:].isdigit():
            return int(part[1:])
    return None

# Directory containing pickle files
root_folder = "/media/DATA1/hunglv/DATA_HOI4D/handpose/handpose/"

data = []
labels = []

temp_pose_list = []  # Temporary list to store pose
temp_label_list = []  # Temporary list to store labels


total_files = sum(1 for subdir, dirs, files in os.walk(root_folder) 
                 for file in files if file.endswith('.txt'))


file_iterator = tqdm(
    [(subdir, file) for subdir, dirs, files in os.walk(root_folder) 
     for file in files if file.endswith('.txt')],
    total=total_files,
    desc="Processing files"
)

# Điều chỉnh đoạn xử lý tệp
total_skipped = 0
skipped_reasons = defaultdict(int)

for subdir, file in file_iterator:
    file_path = os.path.join(subdir, file)
    try:
        label = get_label_from_path(file_path)
        
        if label is None:
            skipped_reasons["No label found"] += 1
            total_skipped += 1
            continue
            
        mapped_label_idx = label_map.get(label, None)
        
        if mapped_label_idx is None:
            skipped_reasons[f"Label {label} not in map"] += 1
            total_skipped += 1
            continue
            
        mapped_label_name = label_name.get(mapped_label_idx, None)
        
        # In ra chi tiết cho một số lượng nhỏ tệp để kiểm tra
        if len(data) < 5 or len(data) % 1000 == 0:
            print(f"Processing: {file_path}")
            print(f"Label: {label}, Mapped idx: {mapped_label_idx}, Name: {mapped_label_name}")

        p = np.loadtxt(file_path).astype('float32')
        p = p.reshape(-1, 3)
        
        for j in range(p.shape[1]):
            p[:, j] = medfilt(p[:, j])
        
        # Sau khi đã áp dụng bộ lọc median, làm phẳng mảng thành 1 chiều
        p_flat = p.flatten()  # Làm phẳng mảng thành 1 chiều với 63 phần tử
        
        # Thêm trực tiếp vào data và labels
        data.append(p_flat)
        labels.append(mapped_label_idx)

    except Exception as e:
        skipped_reasons[f"Error: {str(e)[:50]}"] += 1
        total_skipped += 1
        continue

# In ra thống kê về các tệp bị bỏ qua
print(f"\nTotal files processed: {total_files}")
print(f"Total files added: {len(data)}")
print(f"Total files skipped: {total_skipped}")
print("Skipped reasons:")
for reason, count in skipped_reasons.items():
    print(f"  {reason}: {count}")


output_file = open('/media/DATA1/NAMPV/Hand_recognition/DD_NET/data/HOI4D_random_forest.pkl', 'wb')
print(f"\nSaving data to {output_file}...")
pickle.dump({'data': data, 'labels': labels}, output_file)

