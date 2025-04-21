import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from manopth.manopth.manolayer import ManoLayer
import torch
from collections import defaultdict
from scipy.signal import medfilt
import os


# Định nghĩa label_map
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

# Hàm lấy nhãn từ đường dẫn
def get_label_from_path(path):
    parts = path.split('/')
    for part in parts:
        if part.startswith('C'):
            return int(part[1:])
    return None

# # Hàm đọc và xử lý dữ liệu từ file pickle
# def process_pickle_file(file_path):
#     with open(file_path, 'rb') as f:
#         hand_info = pickle.load(f, encoding='latin1')

#     # Lấy thông tin về theta, beta, trans từ hand_info
#     theta = np.array(hand_info['poseCoeff'])
#     beta = np.array(hand_info['beta'])
#     trans = np.array(hand_info['trans'])

#     # Chuyển đổi thành mảng có thể ghi được và liên tục trong bộ nhớ
#     theta = np.ascontiguousarray(theta)
#     beta = np.ascontiguousarray(beta)
#     trans = np.ascontiguousarray(trans)

#     # Chuyển đổi mảng thành tensor
#     theta_tensor = torch.FloatTensor(theta).unsqueeze(0)
#     beta_tensor = torch.FloatTensor(beta).unsqueeze(0)
#     trans_tensor = torch.FloatTensor(trans).unsqueeze(0)

#     # Bỏ qua cảnh báo về việc tensor không ghi được
#     torch.set_num_threads(1)

#     # Tính toán các đỉnh và khớp của bàn tay
#     hand_verts, hand_joints = manolayer(theta_tensor, beta_tensor)

#     # Chuyển từ tensor về numpy array
#     hand_joints_np = hand_joints.detach().numpy()

#     normalized_hand_joints = np.array([vector / np.linalg.norm(vector) for vector in hand_joints_np])

#     reshaped_hand_joints = normalized_hand_joints.reshape(21, 3)
#     flattened_hand_joints = normalized_hand_joints.flatten()

#     return flattened_hand_joints, get_label_from_path(file_path)

# # Thư mục chứa các file pickle
# root_folder = "/media/DATA_Hieu/DucH/refinehandpose_left/"

# # Khởi tạo MANO model
# manolayer = ManoLayer(
#     mano_root="/media/DATA_Hieu/nampv/DD_NET/manopth/mano/models", use_pca=False, ncomps=45, flat_hand_mean=True, side='right'
# )




# for subdir, dirs, files in os.walk(root_folder):
#     for file in files:
#         file_path = os.path.join(subdir, file)
#         if file.endswith('.pickle'):
#             file_path = os.path.join(subdir, file)
#             try:
#                 hand_joints_1d, label = process_pickle_file(file_path)
#                 mapped_label = label_map.get(label, None)
#                 if mapped_label is not None:
#                     Train['pose'].append(hand_joints_1d.tolist())  # Đảm bảo hand_joints_1d là danh sách
#                     Train['coarse_label'].append(mapped_label)

#                     # Lưu hand_joints_1d ra file txt
#                     output_file_path = os.path.join(output_folder, os.path.splitext(file_path)[0] + '.txt')
#                     print(output_file_path)
#                     with open(output_file_path, 'w') as f:
#                         f.write(" ".join(map(str, hand_joints_1d)))

#             except Exception as e:
#                 print(f"Error processing file {file_path}: {e}")
#                 continue
root_folder = "/media/DATA_Hieu/DucH/"
epsilon = 1e-7

Train = {'pose': [], 'coarse_label': []}
temp_pose_list = []  # Danh sách tạm thời để lưu trữ pose
temp_label_list = []  # Danh sách tạm thời để lưu trữ nhãn

for subdir, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(subdir, file)
            try:
                label = get_label_from_path(file_path)
                mapped_label = label_map.get(label, None)

                txt_file_path = os.path.splitext(file_path)[0] + '.txt'
                p = np.loadtxt(txt_file_path).astype('float32')

                p = np.array(p).reshape(-1, 3)
                p /= 10
                p[:, [1, 2]] = p[:, [2, 1]]
                normalized_hand_joints = np.array([vector / np.linalg.norm(vector) for vector in p])

                normalized_hand_joints = np.ascontiguousarray(normalized_hand_joints)

                pose_tensor = torch.FloatTensor(normalized_hand_joints).unsqueeze(0)

                hand_joints_np = pose_tensor.detach().numpy()

                flattened_hand_joints = hand_joints_np.flatten()

                # if len(flattened_hand_joints) == 63:
                #     temp_pose_list.append(np.array(flattened_hand_joints))  # Lưu trữ pose vào danh sách tạm thời
                # else:
                #     print(f"Bỏ qua mẫu trong file {file_path} có {len(flattened_hand_joints)} giá trị thay vì 63 giá trị mong đợi.")
                temp_pose_list.append(np.array(flattened_hand_joints))  # Lưu trữ pose vào danh sách tạm thời
                temp_label_list.append(mapped_label)  # Lưu trữ nhãn vào danh sách tạm thời

                if len(temp_pose_list) == 32:  # Nếu đã đủ 32 pose
                    # Kiểm tra xem tất cả nhãn có giống nhau không
                    if all(label == temp_label_list[0] for label in temp_label_list):
                        Train['pose'].append(temp_pose_list)  # Thêm tất cả 20 pose vào Train['pose']
                        Train['coarse_label'].append(temp_label_list[0])  # Thêm nhãn vào Train['coarse_label']

                    # Đặt lại danh sách tạm thời
                    temp_pose_list = []
                    temp_label_list = []

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

# Nếu danh sách tạm thời còn sót lại pose sau khi hoàn thành vòng lặp, bỏ qua chúng
if len(temp_pose_list) != 0:
    print("Remaining poses that do not form a complete set of 20 are discarded.")

#Lưu dữ liệu vào file pickle mới
output_file = '/media/DATA_Hieu/nampv/DD_NET/data/HOI4D_Son.pkl'
pickle.dump(Train, open(output_file, "wb"))

# Kiểm tra số lượng label và số lượng pose của mỗi label sau khi ánh xạ
label_count = defaultdict(int)
for label in Train['coarse_label']:
    label_count[label] += 1

# In kết quả
print("Số lượng label khác nhau:", len(label_count))
for label, count in label_count.items():
    print(f"Label {label}: {count} pose")
