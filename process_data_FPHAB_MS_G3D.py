import os
import numpy as np
import pickle
from tqdm import tqdm
import argparse
from scipy.signal import medfilt

label_map = {
    "wirte": 0,
    "wash_sponge": 1,
    "use_flash": 2,
    "unfold_glasses": 3,
    "use_calculator": 4,
    "toast_wine": 5,
    "tear_paper": 6,
    "take_letter_from_enveloppe": 7,
    "stir": 8,
    "squeeze_sponge": 9,
    "squeeze_paper": 10,
    "sprinkle": 11,
    "scratch_sponge": 12,
    "scoop_spoon": 13,
    "receive_coin": 14,
    "read_letter": 15,
    "put_tea_bag": 16,
    "put_sugar": 17,
    "put_salt": 18,
    "prick": 19,
    "pour_wine": 20,
    "pour_milk": 21,
    "pour_liquid_soap": 22,
    "pour_juice_bottle": 23,
    "open_wallet": 24,
    "open_soda_can": 25,
    "open_peanut_butter": 26,
    "open_milk": 27,
    "open_liquid_soap": 28,
    "open_letter": 29,
    "open_juice_bottle": 30,
    "light_candle": 31,
    "high_five": 32,
    "handshake": 33,
    "give_coin": 34,
    "give_card": 35,
    "flip_sponge": 36,
    "flip_pages": 37,
    "drink_mug": 38,
    "close_peanut_butter": 39,
    "close_milk": 40,
    "close_liquid_soap": 41,
    "close_juice_bottle": 42,
    "clean_glasses": 43,
    "charge_cell_phone": 44
}

def get_label_path(path):
    parent_folder = os.path.basename(os.path.dirname(path))
    return parent_folder

def process_folder(data_path, frame_num=32):
    sample_names = []
    sample_labels = []
    batch_data = []

    for label_folder in sorted(os.listdir(data_path)):
        label_path = os.path.join(data_path, label_folder)
        if not os.path.isdir(label_path):
            continue
        
        all_files = sorted([f for f in os.listdir(label_path) if f.endswith(".txt")])
        sequences = [all_files[i:i+frame_num] for i in range(0, len(all_files) - frame_num + 1, frame_num)]

        for seq in tqdm(sequences, desc=f"Processing {label_folder}"):
            data_seq = []
            for file in seq:
                file_path = os.path.join(label_path, file)
                try:
                    data = np.loadtxt(file_path).reshape(21, 3).astype(np.float32)
                    for j in range(3):
                        data[:, j] = medfilt(data[:, j])
                    data_seq.append(data)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    break

            if len(data_seq) == frame_num:
                data_seq = np.stack(data_seq)  # (T, V, C)
                data_seq = np.transpose(data_seq, (2, 0, 1))  # (C, T, V)
                data_seq = np.expand_dims(data_seq, axis=-1)  # (C, T, V, M)
                batch_data.append(data_seq)

                label_name = get_label_path(seq[0])
                mapped_label = label_map.get(label_name, -1)
                sample_labels.append(mapped_label)
                sample_names.append(seq[0])

    all_data = np.array(batch_data).astype(np.float32)  # (N, C, T, V, M)

    return all_data, sample_labels, sample_names

def gendata(data_path, out_path, split='train', frame_num=32):
    out_data_path = os.path.join(out_path, f'{split}_data_joint.npy')
    out_label_path = os.path.join(out_path, f'{split}_label.pkl')

    data_dir = os.path.join(data_path, split)
    data, labels, names = process_folder(data_dir, frame_num)

    print(f"{split.capitalize()} data shape: {data.shape}")
    np.save(out_data_path, data)

    with open(out_label_path, 'wb') as f:
        pickle.dump((names, labels), f)

    print(f"✅ {split.capitalize()} data saved to {out_data_path} and {out_label_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand Pose Data Converter.')
    parser.add_argument('--data_path', default='/media/DATA2/NAMPV/Hand_recognition/DD_NET/data_ista_simplehand_fphab/')
    parser.add_argument('--out_folder', default='/media/DATA2/NAMPV/Hand_recognition/DD_NET/data_ista_simplehand_fphab/')
    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    gendata(args.data_path, args.out_folder, split='train')
    gendata(args.data_path, args.out_folder, split='val')
