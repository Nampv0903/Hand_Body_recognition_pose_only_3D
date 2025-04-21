import os
import numpy as np

# Đường dẫn tới file train_gestures.txt và test_gestures.txt
data_dir = "/media/tth/nampv/"
train_file_path = os.path.join(data_dir, 'train_gestures.txt')
test_file_path = os.path.join(data_dir, 'test_gestures.txt')

# Đọc dữ liệu từ file train_gestures.txt
with open(train_file_path, 'r') as f:
    lines = f.readlines()
    data = []
    for line in lines:
        # Chuyển đổi các phần tử trong mỗi dòng thành số nguyên hoặc chuỗi
        elements = line.strip().split()
        elements = [int(x) if x.isdigit() else x for x in elements]
        data.append(elements)

# Chuyển đổi danh sách thành mảng numpy
data = np.array(data)

# Tính số lượng mẫu cần tách ra
num_samples = len(data)
num_test_samples = int(0.3 * num_samples)  # 30% của tổng số mẫu

# Tạo một mảng indices chứa các chỉ số ngẫu nhiên để chọn mẫu cho tập test
indices = np.random.permutation(num_samples)
test_indices = indices[:num_test_samples]
train_indices = indices[num_test_samples:]

# Tách các mẫu test và train từ dữ liệu ban đầu
test_data = data[test_indices]
train_data = data[train_indices]

# Ghi dữ liệu vào file test_gestures.txt
with open(test_file_path, 'w') as f:
    for row in test_data:
        # Ghi mỗi dòng với các phần tử được phân cách bởi dấu cách
        f.write(' '.join(str(x) for x in row) + '\n')

# Ghi lại dữ liệu còn lại vào file train_gestures.txt
with open(train_file_path, 'w') as f:
    for row in train_data:
        # Ghi mỗi dòng với các phần tử được phân cách bởi dấu cách
        f.write(' '.join(str(x) for x in row) + '\n')

print(f"Đã tạo file test_gestures.txt với {num_test_samples} mẫu ngẫu nhiên từ train_gestures.txt")
print(f"Cập nhật lại file train_gestures.txt với {len(train_data)} mẫu còn lại")
