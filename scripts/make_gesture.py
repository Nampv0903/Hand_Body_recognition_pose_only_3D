import os

# Danh sách các nhãn
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

# Đường dẫn tới thư mục chứa các thư mục Subject_1
root_dir = "/media/tth/DATA_HOI4D/hand_3D_annotation/"

# Kiểm tra xem thư mục root_dir có tồn tại không
if not os.path.exists(root_dir):
    print(f"Thư mục không tồn tại: {root_dir}")
    exit(1)

# Tạo danh sách để lưu trữ thông tin
gesture_info_list = []

# Bản đồ nhãn để chuyển từ Cxx sang chỉ số
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

# Tạo tập hợp để lưu trữ các file đã đọc
read_files = set()

# Lặp qua từng nhãn
for label in labels:
    label_path = os.path.join(root_dir)
    depth_path = os.path.join(label_path, 'refinehandpose_left_new')
    
    if os.path.exists(depth_path) and os.path.isdir(depth_path):
        # Lấy danh sách các tệp tin trong thư mục refinehandpose_left_new
        depth_files = [f for f in os.listdir(depth_path) if os.path.isfile(os.path.join(depth_path, f))]
        
        # Thêm thông tin vào danh sách
        for file_name in depth_files:
            if file_name not in read_files: # Kiểm tra xem file đã được đọc chưa
                read_files.add(file_name) # Thêm file vào tập hợp read_files

                # Tách các phần từ tên tệp
                parts = file_name.split('_')
                if len(parts) > 1:
                    try:
                        gesture_id = parts[0]  # Lấy số cuối của id gesture
                        original_label_id = int(parts[2][1:])  # Bỏ "C" khỏi id label và chuyển sang int
                        if original_label_id in label_map:
                            mapped_label_id = label_map[original_label_id]  # Lấy chỉ số đã được ánh xạ
                            gesture_info_list.append(f"{gesture_id} {mapped_label_id}")  # Ghi id gesture và thứ tự của nhãn

                    except ValueError:
                        print(f"Không thể chuyển đổi label_id: {parts[2][1:]}")
                    except IndexError:
                        print(f"label_id {original_label_id} vượt quá phạm vi của danh sách labels")

# Ghi thông tin vào file train_gestures.txt
output_file_path = "/media/tth/nampv/train_gestures.txt"
count = 0
with open(output_file_path, 'a') as f:
    for info in gesture_info_list:
        count = count + 1
        f.write(info + '\n')

print(f"File train_gestures.txt đã được tạo tại {output_file_path} và có {count} mẫu")
