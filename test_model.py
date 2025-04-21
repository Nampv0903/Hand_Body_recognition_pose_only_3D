import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Lambda, Reshape
import numpy as np
import scipy.ndimage.interpolation as inter
from tqdm import tqdm
import pickle
from scipy.signal import medfilt
from scipy.spatial.distance import cdist
import tensorflow as tf
import random
from scipy.spatial.distance import cdist
import os
import cv2

#========================================================================TEST MODEL==============================================================================
random.seed(1234)

class Config():
    def __init__(self):
        self.frame_l = 32 # the length of frames
        self.joint_n = 21 # the number of joints
        self.joint_d = 3 # the dimension of joints
        self.clc_num = 16 # the number of coarse class
        self.feat_d = 210
        self.filters = 64
C = Config()

def zoom(p,target_l=64,joints_num=25,joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l,joints_num,joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:,m,n] = medfilt(p[:,m,n],3)
            p_new[:,m,n] = inter.zoom(p[:,m,n],target_l/l)[:target_l]
    return p_new

def sampling_frame(p,C):
    full_l = p.shape[0] # full length
    if random.uniform(0,1)<0.5: # aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        s = random.randint(0, full_l-int(valid_l))
        e = s+valid_l # sample end point
        p = p[int(s):int(e),:,:]
    else: # without aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        index = np.sort(np.random.choice(range(0,full_l),int(valid_l),replace=False))
        p = p[index,:,:]
    p = zoom(p,C.frame_l,C.joint_n,C.joint_d)
    return p

def get_CG(p,C):
    M = []
    iu = np.triu_indices(C.joint_n,1,C.joint_n)
    for f in range(C.frame_l):
        #distance max
        d_m = cdist(p[f],np.concatenate([p[f],np.zeros([1,C.joint_d])]),'euclidean')
        d_m = d_m[iu]
        M.append(d_m)
    M = np.stack(M)
    return M

def normlize_range(p):
    # normolize to start point, use the center for hand case
    p[:,:,0] = p[:,:,0]-np.mean(p[:,:,0])
    p[:,:,1] = p[:,:,1]-np.mean(p[:,:,1])
    p[:,:,2] = p[:,:,2]-np.mean(p[:,:,2])
    return p

# Hàm để chuẩn bị dữ liệu cho một pose
def prepare_pose(pose, config):
    pose = np.copy(pose).reshape([-1, 21, 3])
    pose = zoom(pose, target_l=config.frame_l, joints_num=config.joint_n, joints_dim=config.joint_d)
    pose = normlize_range(pose)
    M = get_CG(pose, config)
    return M, pose

# Định nghĩa hàm poses_diff và các hàm liên quan
def poses_diff(x):
    H, W = x.get_shape()[1], x.get_shape()[2]
    x = tf.subtract(x[:, 1:, ...], x[:, :-1, ...])
    x = tf.image.resize(x, size=[H, W])
    return x

def pose_motion(P, frame_l):
    P_diff_slow = Lambda(lambda x: poses_diff(x))(P)
    P_diff_slow = Reshape((frame_l, -1))(P_diff_slow)
    P_fast = Lambda(lambda x: x[:, ::2, ...])(P)
    P_diff_fast = Lambda(lambda x: poses_diff(x))(P_fast)
    P_diff_fast = Reshape((int(frame_l / 2), -1))(P_diff_fast)
    return P_diff_slow, P_diff_fast

# Tải model đã lưu với các hàm tùy chỉnh
DD_Net = load_model('/media/DATA_Hieu/nampv/DD_NET/model/DD_Net_model_37.h5', custom_objects={'poses_diff': poses_diff, 'pose_motion': pose_motion}, compile=False)

# Load bộ dữ liệu kiểm thử
Test = pickle.load(open("/media/DATA_Hieu/nampv/DD_NET/test_model_hoi4d.pkl", "rb"))

# Chọn một pose bất kỳ từ bộ dữ liệu kiểm thử
index = np.random.randint(0, len(Test['pose']))
pose = Test['pose'][index]
name = Test['name_file'][index]
# Chuẩn bị dữ liệu cho pose đó
M, prepared_pose = prepare_pose(pose, C)


# print(M)
# print(prepared_pose)
# exit()

# Dự đoán nhãn cho pose đó
M = np.expand_dims(M, axis=0)
prepared_pose = np.expand_dims(prepared_pose, axis=0)
prediction = DD_Net.predict([M, prepared_pose])

# Nhãn cho các lớp
labels = [
    "Toy Car",            # 1
    "Mug",                # 2
    "Laptop",             # 3
    "Storage Furniture",  # 4
    "Bottle",             # 5
    "Safe",               # 6
    "Bowl",               # 7
    "Bucket",             # 8
    "Scissors",           # 9
    "Pliers",             # 10 (C11)
    "Kettle",             # 11 (C12)
    "Knife",              # 12 (C13)
    "Trash Can",          # 13 (C14)
    "Lamp",               # 14 (C17)
    "Stapler",            # 15 (C18)
    "Chair"               # 16 (C20)
]

# Hiển thị kết quả
predicted_label_index = np.argmax(prediction, axis=1)[0]
predicted_label = labels[predicted_label_index]
print(f'Predicted label: {predicted_label}')
print(prediction)
print(prepared_pose)
print(name)
# Hiển thị nhãn thực sự
actual_label_index = Test['coarse_label'][index] - 1
actual_label = labels[actual_label_index]
print(f'Actual label: {actual_label}')
