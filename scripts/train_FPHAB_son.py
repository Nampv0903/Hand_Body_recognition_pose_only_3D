import numpy as np
import math
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import glob
from tqdm import tqdm
import pickle
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt
from scipy.spatial.distance import cdist

import scipy.ndimage.interpolation as inter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import *


random.seed(1234)

class Config():
    def __init__(self):
        self.frame_l = 32 # the length of frames
        self.joint_n = 21 # the number of joints
        self.joint_d = 3 # the dimension of joints
        self.clc_num = 45 # the number of coarse class
        self.feat_d = 210
        self.filters = 64
C = Config()

# Temple resizing function
###################################################################################
#Rescale to be 64 frames
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

from scipy.spatial.distance import cdist
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

def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(8,8)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f' % (p)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f' % (p)
    
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cbar=False, cmap="YlGnBu")
    plt.savefig(filename)
    

    precision_per_label = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_per_label = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_label = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    

    log_filename = filename.replace('.png', '_metrics.log')
    with open(log_filename, 'w') as log_file:
        log_file.write('Label | Precision | Recall | F1-score\n')
        log_file.write('-' * 35 + '\n')
        for label, p, r, f1 in zip(labels, precision_per_label, recall_per_label, f1_per_label):
            log_file.write(f'{label:<30} | {p:.4f} | {r:.4f} | {f1:.4f}\n')
        
     
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        log_file.write('\nOverall Metrics:\n')
        log_file.write(f'Precision: {precision:.4f}\n')
        log_file.write(f'Recall: {recall:.4f}\n')
        log_file.write(f'F1-score: {f1:.4f}\n')
    
def poses_diff(x):
    H, W = x.get_shape()[1],x.get_shape()[2]
    x = tf.subtract(x[:,1:,...],x[:,:-1,...])
    x = tf.image.resize(x,size=[H,W])
    return x

def pose_motion(P,frame_l):
    P_diff_slow = Lambda(lambda x: poses_diff(x))(P)
    P_diff_slow = Reshape((frame_l,-1))(P_diff_slow)
    P_fast = Lambda(lambda x: x[:,::2,...])(P)
    P_diff_fast = Lambda(lambda x: poses_diff(x))(P_fast)
    P_diff_fast = Reshape((int(frame_l/2),-1))(P_diff_fast)
    return P_diff_slow,P_diff_fast

def c1D(x,filters,kernel):
    x = Conv1D(filters, kernel_size=kernel,padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def block(x,filters):
    x = c1D(x,filters,3)
    x = c1D(x,filters,3)
    return x

def d1D(x,filters):
    x = Dense(filters,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def build_FM(frame_l=32, joint_n=22, joint_d=2, feat_d=231, filters=16):
    M = Input(shape=(frame_l, feat_d))
    P = Input(shape=(frame_l, joint_n, joint_d))

    diff_slow, diff_fast = pose_motion(P, frame_l)

    x = c1D(M, filters * 2, 1)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x, filters, 3)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x, filters, 1)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x_d_slow = c1D(diff_slow, filters * 2, 1)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow, filters, 3)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow, filters, 1)
    x_d_slow = MaxPool1D(2)(x_d_slow)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)

    x_d_fast = c1D(diff_fast, filters * 2, 1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast, filters, 3)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast, filters, 1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)

    x = concatenate([x, x_d_slow, x_d_fast])
    x = block(x, filters * 2)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x, filters * 4)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x, filters * 8)
    x = SpatialDropout1D(0.1)(x)

    return Model(inputs=[M, P], outputs=x)


def build_DD_Net(C):
    # Define inputs
    M = Input(name='M', shape=(C.frame_l, C.feat_d))
    P = Input(name='P', shape=(C.frame_l, C.joint_n, C.joint_d))

    # Build feature module FM
    FM = build_FM(C.frame_l, C.joint_n, C.joint_d, C.feat_d, C.filters)

    # Pass inputs through FM
    x = FM([M, P])

    # Global max pooling
    x = GlobalMaxPool1D()(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output layer for classification
    x = Dense(C.clc_num, activation='softmax')(x)

    # Create model
    model = Model(inputs=[M, P], outputs=x)
    return model
    
DD_Net = build_DD_Net(C)
DD_Net.summary()
Train= pickle.load(open("/media/DATA2/NAMPV/Hand_recognition/DD_NET/data/FPHAB_simplehand_data_train.pkl", "rb"))
Test = pickle.load(open("/media/DATA2/NAMPV/Hand_recognition/DD_NET/data/FPHAB_simplehand_data_test.pkl", "rb"))

X_0 = []
X_1 = []
Y = []
for i in tqdm(range(len(Train['pose']))):
    p = np.copy(Train['pose'][i]).reshape([-1,21,3])
    p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)
    p = normlize_range(p)

    label = np.zeros(C.clc_num)
    label[Train['coarse_label'][i]-1] = 1

    M = get_CG(p,C)

    X_0.append(M)
    X_1.append(p)
    Y.append(label)
print(len(Train['pose']))  # Check the length of X_0
print(X_0)  # Check the actual contents of X_0

X_0 = np.stack(X_0)
X_1 = np.stack(X_1)
Y = np.stack(Y)

X_test_0 = []
X_test_1 = []
Y_test = []
for i in tqdm(range(len(Test['pose']))):
    p = np.copy(Test['pose'][i]).reshape([-1,21,3])
    p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)
    p = normlize_range(p)

    label = np.zeros(C.clc_num)
    label[Test['coarse_label'][i]-1] = 1

    M = get_CG(p,C)

    X_test_0.append(M)
    X_test_1.append(p)
    Y_test.append(label)

X_test_0 = np.stack(X_test_0)
X_test_1 = np.stack(X_test_1)
Y_test = np.stack(Y_test)

lr = 1e-4
DD_Net.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(lr),metrics=['accuracy'])
lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, cooldown=5, min_lr=5e-5)
history = DD_Net.fit([X_0,X_1],Y,
            batch_size=16,
            epochs=500,
            verbose=True,
            shuffle=True,
            callbacks=[lrScheduler],
            validation_data=([X_test_0,X_test_1],Y_test)
            )

DD_Net.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(lr),metrics=['accuracy'])
lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, cooldown=5, min_lr=5e-5)
history = DD_Net.fit([X_0,X_1],Y,
            batch_size=16,
            epochs=500,
            verbose=True,
            shuffle=True,
            callbacks=[lrScheduler],
            validation_data=([X_test_0,X_test_1],Y_test)
            )

            
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('/media/DATA2/NAMPV/Hand_recognition/DD_NET/result/loss_FPHAB_simplehand.png')

DD_Net.save('/media/DATA2/NAMPV/Hand_recognition/DD_NET/model/FPHAB/DD_Net_FPHAB_simplehand_model.keras')

Y_pred = DD_Net.predict([X_test_0,X_test_1])

labels = [ "wirte",
    "wash_sponge",
    "use_flash",
    "use_calculator",
    "unfold_glasses",
    "toast_wine",
    "tear_paper",
    "take_letter_from_enveloppe",
    "stir",
    "squeeze_sponge",  
    "squeeze_paper", 
    "sprinkle",  
    "scratch_sponge", 
    "scoop_spoon",  
    "receive_coin",  
    "read_letter",
    "put_tea_bag",
    "put_sugar",
    "put_salt",
    "prick",
    "pour_wine",
    "pour_milk",
    "pour_liquid_soap",
    "pour_juice_bottle",
    "open_wallet",
    "open_soda_can",
    "open_peanut_butter",
    "open_milk",
    "open_liquid_soap",
    "open_letter",
    "open_juice_bottle",
    "light_candle",
    "high_five",
    "handshake",
    "give_coin",
    "give_card",
    "flip_sponge",
    "flip_pages",
    "drink_mug",
    "close_peanut_butter",
    "close_milk",
    "close_liquid_soap",
    "close_juice_bottle",
    "clean_glasses",
    "charge_cell_phone" 
    ]


y_true = []
for i in np.argmax(Y_test,axis=1):
    y_true.append(labels[i])

y_pred = []
for i in np.argmax(Y_pred,axis=1):
    y_pred.append(labels[i])

cm_analysis(y_true,y_pred, '/media/DATA2/NAMPV/Hand_recognition/DD_NET/result/FPHAB_simplehand.png', labels, ymap=None, figsize=(16,16))