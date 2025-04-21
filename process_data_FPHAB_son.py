# N T M V C
# pkl
import glob
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
label_map = {
    "wirte": 1,
    "wash_sponge": 2,
    "use_flash": 3,
    "use_calculator": 5,
    "unfold_glasses": 4,
    "toast_wine": 6,
    "tear_paper": 7,
    "take_letter_from_enveloppe": 8,
    "stir": 9,
    "squeeze_sponge": 10,  
    "squeeze_paper": 11, 
    "sprinkle": 12,  
    "scratch_sponge": 13, 
    "scoop_spoon": 14,  
    "receive_coin": 15,  
    "read_letter": 16,
    "put_tea_bag":17,
    "put_sugar":18,
    "put_salt":19,
    "prick":20,
    "pour_wine":21,
    "pour_milk":22,
    "pour_liquid_soap":23,
    "pour_juice_bottle":24,
    "open_wallet":25,
    "open_soda_can":26,
    "open_peanut_butter":27,
    "open_milk":28,
    "open_liquid_soap":29,
    "open_letter":30,
    "open_juice_bottle":31,
    "light_candle":32,
    "high_five":33,
    "handshake":34,
    "give_coin":35,
    "give_card":36,
    "flip_sponge":37,
    "flip_pages":38,
    "drink_mug":39,
    "close_peanut_butter":40,
    "close_milk":41,
    "close_liquid_soap":42,
    "close_juice_bottle":43,
    "clean_glasses":44,
    "charge_cell_phone":45
}

#def get_label_path(path):

 #       stringer=path
  #      label=stringer[9:-10]
   #     return label
        
def get_label_path(path):
    parent_folder = os.path.basename(os.path.dirname(path))
    return parent_folder

# def get_label_from_path(path):
#     parts = path.split('/')
#     for part in parts:
#         if part.startswith('C') and part[1:].isdigit():
#             return int(part[1:])
#     return None

# Directory containing pickle files
root_folder = "/media/DATA1/NAMPV/Hand_pose_3d/FPHAB_Hamuco_pred/"

Train = {'pose': [], 'coarse_label': []}
Test = {'pose': [], 'coarse_label': []}
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

for subdir, file in file_iterator:

    file_path = os.path.join(subdir, file)
    
    try:
        labels = get_label_path(file_path)

        mapped_label = label_map.get(labels, None)

        if mapped_label is None:
            file_iterator.set_postfix_str(f"Skipped: {file} (invalid label)")
            continue
        result = "/".join(file_path.rsplit("/", 2)[:-1]) + "/"
      
        count = len(glob.glob(os.path.join(result, "*")))
        p = np.loadtxt(file_path).astype('float32')

        p = p.reshape(-1, 3)
        

        for j in range(p.shape[1]):
            p[:, j] = medfilt(p[:, j])

        temp_pose_list.append(p)
        temp_label_list.append(mapped_label)
        
        if len(temp_pose_list) == 32:
            random_number = random.choice([0,1,2,3,4,5,6,7,8,9])


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
        print("---loi roi---")
        continue


if len(temp_pose_list) != 0:
    print(f"Discarded {len(temp_pose_list)} remaining poses that do not form a complete set of 32")


train_out = '/media/DATA2/NAMPV/Hand_recognition/DD_NET/data/FPHAB_hamuco_data_train.pkl'
print(f"\nSaving data to {train_out}...")
pickle.dump(Train, open(train_out, "wb"))

test_out = '/media/DATA2/NAMPV/Hand_recognition/DD_NET/data/FPHAB_hamuco_data_test.pkl'
print(f"\nSaving data to {test_out}...")
pickle.dump(Test, open(test_out, "wb"))

label_count = defaultdict(int)
for label in Train['coarse_label']:
    label_count[label] += 1
label_count = defaultdict(int)
for label in Test['coarse_label']:
    label_count[label] += 1
