import os
import cv2
from glob import glob
from tqdm import tqdm
from utils import crop_face_from_id

def crop_and_resize_images_in_folder():
    source_dir = "/home/bits/Desktop/faceparsing1"
    target_dir = "/home/bits/Desktop/faceparsing2"

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # faceparsing1 폴더 내의 모든 하위 폴더를 순회
    all_subfolders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    folder_counter = 0  # 폴더 이름을 위한 카운터 (0부터 시작)
    for folder_name in tqdm(all_subfolders):
        source_folder_path = os.path.join(source_dir, folder_name)
        target_folder_path = os.path.join(target_dir, str(folder_counter))
        folder_counter += 1

        if not os.path.exists(target_folder_path):
            os.mkdir(target_folder_path)

        image_paths = glob(os.path.join(source_folder_path, "*.png"))

        file_counter = 1  # 파일 이름을 위한 카운터 (1부터 시작)
        for img_path in image_paths:
            img = cv2.imread(img_path)
            try:
                cropped_face = crop_face_from_id(img, weight_path="/home/bits/weight")
            except RuntimeError as e:
                print(f"[Error] {e} in image: {img_path}")
                continue

            if cropped_face is None or cropped_face.size == 0:
                print(f"Failed to crop face or invalid image from {img_path}")
                continue

            resized_face = cv2.resize(cropped_face, (112, 112))
            target_img_path = os.path.join(target_folder_path, f"{file_counter}.jpg")
            cv2.imwrite(target_img_path, resized_face)
            file_counter += 1

if __name__ == "__main__":
    crop_and_resize_images_in_folder()

