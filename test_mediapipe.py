import numpy as np
import cv2
import mediapipe as mp
import os

def read_hand_data(file_path):
    """Đọc dữ liệu bàn tay từ file và trả về numpy array."""
    with open(file_path, 'r') as file:
        data = file.read().split()
    return np.array(data, dtype=float).reshape(-1, 3)

def draw_hand(landmarks, image, drawing_spec=None):
    """Vẽ các điểm và kết nối bàn tay lên hình ảnh."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    if drawing_spec is None:
        drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

    hand_landmarks = mp_hands.HandLandmark
    connections = mp_hands.HAND_CONNECTIONS

    height, width, _ = image.shape

    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]

        # Chuyển đổi từ chuẩn hóa thành tọa độ pixel
        start_point = (int(start_point[0] * width), int(start_point[1] * height))
        end_point = (int(end_point[0] * width), int(end_point[1] * height))

        cv2.line(image, start_point, end_point, drawing_spec.color, drawing_spec.thickness)

    for idx, landmark in enumerate(landmarks):
        x, y = int(landmark[0] * width), int(landmark[1] * height)
        cv2.circle(image, (x, y), drawing_spec.thickness, drawing_spec.color, -1)

    return image

def draw_label(image, label):
    """Vẽ label lên góc trái màn hình."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 0, 0)  # Màu chữ
    thickness = 3
    line_type = cv2.LINE_AA

    # Xác định vị trí vẽ label
    position = (20, 30)

    # Vẽ label lên hình ảnh
    cv2.putText(image, label, position, font, font_scale, font_color, thickness, line_type)

    return image

def main():
    # Đường dẫn đến file txt chứa dữ liệu bàn tay
    hand_file_path = '/media/DATA_Hieu/nampv/DD_NET/txt17/63.txt'  
    # Đường dẫn đến hình ảnh để phát hiện bàn tay
    image_file_path = '/media/DATA_Hieu/nampv/DD_NET/frames/C17/63.png'
    output_dir = 'output'  # Thư mục để lưu ảnh đầu ra
    output_filename = 'hand_image63.png'  # Tên file ảnh đầu ra
    label = 'Lamp'  # Label muốn hiển thị

    # Tạo thư mục output nếu nó chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Đọc dữ liệu bàn tay từ file
    hand_data = read_hand_data(hand_file_path)

    # Đọc hình ảnh từ file
    image = cv2.imread(image_file_path)
    if image is None:
        raise FileNotFoundError(f"Không thể đọc hình ảnh từ {image_file_path}")

    height, width, _ = image.shape

    # Sử dụng Mediapipe để phát hiện bàn tay
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils

    # Chuyển đổi hình ảnh sang RGB cho Mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # Vẽ các bàn tay phát hiện được từ Mediapipe
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
            # Vẽ bàn tay phát hiện được từ Mediapipe
            image = draw_hand(landmarks, image, drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

    # Chuyển dữ liệu bàn tay từ file txt thành tọa độ pixel và vẽ lên hình ảnh
    # Đảm bảo rằng dữ liệu không bị chuẩn hóa khi vẽ
    # hand_data[:, :2] = (hand_data[:, :2] - hand_data[:, :2].min()) / (hand_data[:, :2].max() - hand_data[:, :2].min())
    # image = draw_hand(hand_data, image, drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

    # Vẽ label lên hình ảnh
    image = draw_label(image, label)

    # Lưu ảnh
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, image)
    print(f"Ảnh đã được lưu tại: {output_path}")

if __name__ == "__main__":
    main()
