import cv2
import os
import numpy as np
import bcolz



def images_to_bcolz(image_folder, output_path):
    subfolders = sorted([d for d in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, d))])

    target_size = (112, 112)  # 원하는 이미지 크기

    # 데이터 shape 설정
    data_shape = (
    len(subfolders) * len(os.listdir(os.path.join(image_folder, subfolders[0]))), 3, target_size[1], target_size[0])

    # bcolz array 생성
    image_array = bcolz.zeros(data_shape, dtype=np.float32, mode='w', rootdir=os.path.join(output_path, 'dataset'),
                              chunklen=13)

    idx = 0
    for subfolder in subfolders:
        for image_file in os.listdir(os.path.join(image_folder, subfolder)):
            img_path = os.path.join(image_folder, subfolder, image_file)
            img = cv2.imread(img_path)  # OpenCV로 이미지 읽기

            img_resized = cv2.resize(img, target_size)  # 이미지 리사이즈
            img_array = np.transpose(img_resized, (2, 0, 1))  # (높이, 너비, 채널)을 (채널, 높이, 너비)로 변경

            # [1, -1] 범위로 정규화
            img_array = (img_array / 255.0) * 2 - 1

            image_array[idx] = img_array
            idx += 1

    image_array.flush()  # 디스크에 변경 사항 저장

image_folder = '/home/bits/FaceID/data/kface1_val/'  # 변환할 이미지 폴더
output_path = '/home/bits/arcface-tf2/data/test_dataset/kface1/'  # bcolz로 저장할 경로
images_to_bcolz(image_folder, output_path)
