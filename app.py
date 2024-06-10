from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from gevent.pywsgi import WSGIServer
import io
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)
CORS(app)  # Cho phép tất cả các nguồn


def extract_surf_features(frame, max_kpt):
    # Khởi tạo bộ trích xuất đặc trưng SURF
    sirf = cv2.SIFT_create(nfeatures=max_kpt)
    # Tìm key points và descriptors của ảnh
    keypoints, descriptors = sirf.detectAndCompute(frame, None)
    print(descriptors.shape)
    #Nếu đặc trưng là None
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)

    # Nếu số lượng keypoint nhiều hơn 1000, thì lấy 1000 kpt đầu tiên
    if len(keypoints) > max_kpt:
        keypoints = keypoints[:max_kpt]
        descriptors = descriptors[:max_kpt, :]
    
    # Nếu số lượng keypoints ít hơn max_kpt, thêm các số 0 vào các descriptors
    if len(keypoints) < max_kpt:
        pad_width = ((0, max_kpt - len(keypoints)), (0, 0))
        descriptors = np.pad(descriptors, pad_width, mode='constant')

    return keypoints, descriptors

def read_video(name_video, video_path, max_kpt = 1000, frame_skip = 24):
    
    if os.path.exists(f'feature_new/{name_video}.npy'):
        return
    # Mở video
    cap = cv2.VideoCapture(video_path)
    # Đọc chiều dài và chiều rộng của video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f'Chiều cao: {height}, chiều rộng {width} ')
    
    all_descriptors = []
    # Kiểm tra xem video có mở thành công không
    if not cap.isOpened():
        print("Không thể mở video.")
        exit()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip == 0:
            # Trích xuất đặc trưng từ mỗi frame
            keypoints, descriptors = extract_surf_features(frame, max_kpt)
            # print(descriptors.shape)
            # Hiển thị số lượng key points tìm thấy trong mỗi frame
            # print(f"Số lượng key points trong frame: {len(keypoints)}")
            all_descriptors.append(descriptors)
            # Hiển thị frame với key points
            frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None)
            cv2.imshow('Frame', frame_with_keypoints)

        # Đợi 25ms, bấm phím 'q' để thoát
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(frame_count)
   
    # np.save(f'feature_old/{name_video}.npy', np.array(all_descriptors))


def read_files_in_directory(directory):
    # Kiểm tra xem thư mục tồn tại không
    if not os.path.exists(directory):
        print("Thư mục không tồn tại.")
        return
    
    # Kiểm tra xem 'directory' là một thư mục không
    if not os.path.isdir(directory):
        print(f"{directory} không phải là một thư mục.")
        return
    
    # Lặp qua tất cả các tệp trong thư mục và in ra tên của chúng
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # filepath2 = os.path.join(filepath, filename)
        # for video_file in os.listdir(filepath):
        # video_path = os.path.join(filepath, video_file) # đường dẫn file
        print(filepath) 
        read_video(filename, rf"{filepath}")


# directory_path = r'E:\codeKHANH\video_recognition\Video_Recognition\dataset'
# read_files_in_directory(directory_path) #trích xuất đặc trưng
# read_video('v22_animal.mp4',r'dataset\flowers\flowers\v22_flower.mp4', max_kpt = 1000, frame_skip = 24)

# #################### STAGE 2 ############################

def resize_image(image, width, height):
    # Lấy kích thước hiện tại của ảnh
    # print(image)
    h, w = image.shape[:2]

    # Tính toán tỉ lệ và kích thước mới
    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize ảnh
    resized_image = cv2.resize(image, (new_w, new_h))

    # Tạo ảnh nền với kích thước mới và màu đen
    result_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Tính toán vị trí để đặt ảnh đã resize vào ảnh nền
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2

    # Đặt ảnh đã resize vào giữa ảnh nền
    result_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return result_image


def cosine_similarity(A, B): 
    '''
    Input: 2 vector A, B có cùng chiều
    output: độ tương đồng của 2 vector
    '''
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / (norm_A * norm_B)
    return similarity

def find_similar_videos(query_image, directory, k=3):
    top_videos = []
    #tiền xử lý ảnh video theo chiều dài/ rộng cố định
    img_cropped = resize_image(query_image, 1280, 720)
    #trích xuất đặc trưng của ảnh đầu vào
    kpt, feature_img = extract_surf_features(img_cropped, max_kpt=1000)
    # print(feature_img)
    # feature_img = min_max_normalize(feature_img)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        for video_file in os.listdir(filepath):
            video_feature_path = os.path.join(filepath, video_file)
            print(video_feature_path)
            video_features = np.load(rf"{video_feature_path}")
            # Tính toán độ tương đồng giữa ảnh đầu vào và mỗi frame của 1 video
            similarities = []
            for video_feature in video_features:
                # video_feature = min_max_normalize(video_feature)
                similarity = cosine_similarity(feature_img.flatten(), video_feature.flatten())
                similarities.append(similarity)

            # lấy ra độ tương đồng lớn nhất trong 1 video
            max_similarity_each_video = np.max(similarities)
            # Thêm vào danh sách cặp (điểm số, tên tệp video)
            top_videos.append((max_similarity_each_video, video_file[:-4]))
    # Sắp xếp danh sách theo điểm số và lấy ra 3 cặp đầu tiên
    top_videos.sort(reverse=True)
    # top_3_videos = top_videos[:3]
    # return top_3_videos
    result = str(top_videos[0]) + "-" + str(top_videos[1]) + "-" + str(top_videos[2])
    return result

def convert_result(result):
    arr_result = [str(i) for i in result.split('-')]
    for i in range(3):
        arr_result[i] = arr_result[i].replace('(', '')
        arr_result[i] = arr_result[i].replace(')', '')
        arr_result[i] = arr_result[i].replace('\'', '')
        arr_result[i] = arr_result[i].replace(',', '')
        arr_result[i] = arr_result[i].replace(' ', '-')
    for i in range(3):
        a, b = arr_result[i].split('-')
        x, y = b.split('_')
        y = y.replace('.mp4', '')
        b = "web/assets/dataset/" + y + "s/" + y + "s/" + x + "_" + y + ".mp4"
        c = x + "_" + y + ".mp4"
        arr_result[i] = c + "-" + a + "-" + b
    ans = "-".join(arr_result)
    return ans

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    # if file:
    #     img = Image.open(file.stream)
        
    #     # Xử lý ảnh tại đây, ví dụ chuyển ảnh thành chuỗi
    #     # Đây là phần bạn cần thay đổi theo yêu cầu xử lý ảnh của bạn
    #     result_string = f"Processed image size: kkk {img.size}"
        
    #     return jsonify({'result': result_string})
    if file:
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        directory_feature = 'feature_new'  # Đường dẫn tới thư mục chứa các đặc trưng video
        similar_videos = find_similar_videos(img, directory_feature, k=3)
        # return jsonify({'result': similar_videos})
        similar_videos = convert_result(similar_videos)
        print(similar_videos)
        return str(similar_videos)
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
    app.run(debug=True)
