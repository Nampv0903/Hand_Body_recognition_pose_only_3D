import pickle
from collections import defaultdict


# with open('/media/tth/nampv/data/test.pkl', 'rb') as f:
#     data = pickle.load(f)

with open("/media/DATA2/NAMPV/Hand_recognition/DD_NET/data/FPHAB_data_train_son.pkl", 'rb') as f:
    data = pickle.load(f)


poses = data['pose']
labels = data['coarse_label']
# for pose in poses:
#     for label in labels:
#         print(pose)
#         print(label)
#         exit()
# print(type(poses))
# exit()


label_count = defaultdict(int)

for label in labels:
    label_count[label] += 1


print("so label khac nhau:", len(label_count))
for label, count in label_count.items():
    print(f"Label {label}: {count} pose")



#with open('/media/DATA_Hieu/nampv/DD_NET/pose1.txt', 'w') as f:
#    for pose, label in zip(poses, labels):
#        pose_str = ' '.join(map(str, pose))
#        f.write(f"{pose_str} {label}\n")

