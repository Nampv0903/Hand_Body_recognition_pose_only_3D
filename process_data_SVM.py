# N T M V C
# pkl
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from collections import defaultdict
from scipy.signal import medfilt
import os
from tqdm import tqdm

# Define label_map
label_map = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    11: 10,  # Pliers
    12: 11,  # Kettle
    13: 12,  # Knife
    14: 13,  # Trash Can
    17: 14,  # Lamp
    18: 15,  # Stapler
    20: 16   # Chair
}

label_name = {
    1: "Toy Car",
    2: "Mug",
    3: "Laptop",
    4: "Storage Furniture",
    5: "Bottle",
    6: "Safe",
    7: "Bowl",
    8: "Bucket",
    9: "Scissors",
    10: "Pliers",
    11: "Kettle",
    12: "Knife",
    13: "Trash Can",
    14: "Lamp",
    15: "Stapler",
    16: "Chair"
}

def get_label_from_path(path):
    parts = path.split('/')
    for part in parts:
        if part.startswith('C') and part[1:].isdigit():
            return int(part[1:])
    return None


def landmarks_to_csv_line(landmarks):
 
    flattened = landmarks.flatten()
  
    return ','.join(map(str, flattened))

# Directory containing pickle files
root_folder = "/media/DATA1/hunglv/DATA_HOI4D/handpose/handpose/"


output_csv = '/media/DATA1/NAMPV/Hand_recognition/DD_NET/data/HOI4D_SVM.csv'
print(f"\nSaving data to {output_csv}...")


with open(output_csv, 'w') as file_name:
  
    # file_name.write("# Landmarks (63 values) + Label\n")
  
    total_files = sum(1 for subdir, dirs, files in os.walk(root_folder) 
                    for file in files if file.endswith('.txt'))
    
    file_iterator = tqdm(
        [(subdir, file) for subdir, dirs, files in os.walk(root_folder) 
         for file in files if file.endswith('.txt')],
        total=total_files,
        desc="Processing files"
    )
    

    label_count = defaultdict(int)
    processed_count = 0
    
    for subdir, file in file_iterator:
        file_path = os.path.join(subdir, file)
        try:
            label = get_label_from_path(file_path)
            mapped_label = label_map.get(label, None)
            
            if mapped_label is None:
                file_iterator.set_postfix_str(f"Skipped: {file} (invalid label)")
                continue
                
          
            mapped_label_name = label_name.get(mapped_label, "Unknown")
            
         
            p = np.loadtxt(file_path).astype('float32')
            p = p.reshape(-1, 3)
            
            for j in range(p.shape[1]):
                p[:, j] = medfilt(p[:, j])
            
       
            landmarks_csv = landmarks_to_csv_line(p)
            
       
            line = landmarks_csv + f',"{mapped_label_name}"\n'

            file_name.write(line)
            
       
            label_count[mapped_label] += 1
            processed_count += 1
            
            if processed_count % 100 == 0:
                file_iterator.set_postfix_str(f"Processed: {processed_count}, Current label: {mapped_label_name}")
                
        except Exception as e:
            file_iterator.set_postfix_str(f"Error: {str(e)[:30]}...")
            continue


print("\nSummary of processed files by label:")
for label, count in label_count.items():
    label_name_str = label_name.get(label, "Unknown")
    print(f"  Label {label} ({label_name_str}): {count} files")

print(f"Total processed: {processed_count} files")
print(f"Data saved to {output_csv}")