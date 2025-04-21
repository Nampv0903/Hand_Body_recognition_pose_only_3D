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
import random
# Define label_map
# label_map = {
#     1: 1,
#     2: 2,
#     3: 3,
#     4: 4,
#     5: 5,
#     6: 6,
#     7: 7,
#     8: 8,
#     9: 9,
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

# Directory containing pickle files
root_folder = "/media/DATA2/DATA_HOI4D/handpose/handpose/"

Train = {'pose': [], 'coarse_label': []}
Test = {'pose': [], 'coarse_label': []}
temp_pose_list = []  # Temporary list to store pose
temp_label_list = []  # Temporary list to store labels



total_files = sum(1 for subdir, dirs, files in os.walk(root_folder) 
                 for file in files if file.endswith('.txt'))
print("_____________")
print(total_files)
print("_____________")

file_iterator = tqdm(
    [(subdir, file) for subdir, dirs, files in os.walk(root_folder) 
     for file in files if file.endswith('.txt')],
    total=total_files,
    desc="Processing files"
)

for subdir, file in file_iterator:
    file_path = os.path.join(subdir, file)
    try:
        label = get_label_from_path(file_path)
        mapped_label = label_map.get(label, None)
        
        if mapped_label is None:
            file_iterator.set_postfix_str(f"Skipped: {file} (invalid label)")
            continue

        p = np.loadtxt(file_path).astype('float32')
        p = p.reshape(-1, 3)


        for j in range(p.shape[1]):
            p[:, j] = medfilt(p[:, j])

        temp_pose_list.append(p)
        temp_label_list.append(mapped_label)

        if len(temp_pose_list) == 32:
            random_number = random.choice([0, 1,2,3,4,5,6,7,8,9])
            if  random_number in [1,2,3]:

                if all(label == temp_label_list[0] for label in temp_label_list):
                    Test['pose'].append(temp_pose_list)

                    Test['coarse_label'].append(temp_label_list[0])

                    file_iterator.set_postfix_str(f"Added batch: label {temp_label_list[0]}")
                else:

                    file_iterator.set_postfix_str("Skipped batch: mixed labels")
                temp_pose_list = []
                temp_label_list = []

            else:
                if all(label == temp_label_list[0] for label in temp_label_list):
                    Train['pose'].append(temp_pose_list)
                    Train['coarse_label'].append(temp_label_list[0])
                    file_iterator.set_postfix_str(f"Added batch: label {temp_label_list[0]}")
                else:
                    file_iterator.set_postfix_str("Skipped batch: mixed labels")

                temp_pose_list = []
                temp_label_list = [] 

    except Exception as e:
        file_iterator.set_postfix_str(f"Error: {str(e)[:30]}...")
        continue


if len(temp_pose_list) != 0:
    print(f"Discarded {len(temp_pose_list)} remaining poses that do not form a complete set of 32")


train_out = '/media/DATA2/NAMPV/Hand_recognition/DD_NET/data/HOI4D_FPHAB_data_train_son.pkl'
print(f"\nSaving data to {train_out}...")
pickle.dump(Train, open(train_out, "wb"))

test_out = '/media/DATA2/NAMPV/Hand_recognition/DD_NET/data/HOI4D_FPHAB_data_test_son.pkl'
print(f"\nSaving data to {test_out}...")
pickle.dump(Test, open(test_out, "wb"))

label_count = defaultdict(int)
for label in Train['coarse_label']:
    label_count[label] += 1
label_count = defaultdict(int)
for label in Test['coarse_label']:
    label_count[label] += 1
