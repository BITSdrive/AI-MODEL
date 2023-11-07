import os
import cv2
import numpy as np
import random


def create_lfw_datasets_with_labels_v14(image_dir, bin_filepath, npy_filepath, img_size=(112, 112)):
    folders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    image_data = bytearray()
    issame_list = []

    all_image_sets = []
    for folder in folders:
        folder_path = os.path.join(image_dir, folder)
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                       f.endswith(('.png', '.jpg', '.jpeg'))]
        all_image_sets.append(image_files)

    same_pairs = 0
    different_pairs = 0
    while len(issame_list) < 250:
        folder_index = random.randint(0, len(all_image_sets) - 1)

        if len(all_image_sets[folder_index]) < 2 and same_pairs < 125:
            continue

        if same_pairs < 125:  # Same folder
            img1_file = all_image_sets[folder_index].pop(random.randint(0, len(all_image_sets[folder_index]) - 1))
            img2_file = all_image_sets[folder_index].pop(random.randint(0, len(all_image_sets[folder_index]) - 1))
            same_pairs += 1
        else:  # Different folder
            different_folder_indices = list(range(len(all_image_sets)))
            different_folder_indices.remove(folder_index)
            different_folder_index = random.choice(different_folder_indices)

            if not all_image_sets[different_folder_index]:
                continue

            img1_file = all_image_sets[folder_index].pop(random.randint(0, len(all_image_sets[folder_index]) - 1))
            img2_file = all_image_sets[different_folder_index].pop(
                random.randint(0, len(all_image_sets[different_folder_index]) - 1))
            different_pairs += 1

        img1 = cv2.imread(img1_file)
        img2 = cv2.imread(img2_file)

        img1 = cv2.resize(img1, img_size)
        img2 = cv2.resize(img2, img_size)

        # Convert to RGB format
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Encode as JPEG and append to the image_data
        _, img1_encoded = cv2.imencode('.jpg', img1)
        _, img2_encoded = cv2.imencode('.jpg', img2)
        image_data.extend(img1_encoded.tobytes())
        image_data.extend(img2_encoded.tobytes())

        issame = os.path.dirname(img1_file) == os.path.dirname(img2_file)
        issame_list.append(issame)

        # Remove folders with no images left
        all_image_sets = [img_set for img_set in all_image_sets if img_set]

    # Save the image data
    with open(bin_filepath, 'wb') as f:
        f.write(image_data)

    issame_array = np.array(issame_list)
    np.save(npy_filepath, issame_array)

    return f"Saved image data to {bin_filepath}", f"Saved labels to {npy_filepath}"


# Sample usage
print(create_lfw_datasets_with_labels_v14('/home/bits/FaceID/data/kface1_val/',
                                          '/home/bits/arcface-tf2/data/test_dataset/kface1/kface1/kface.bin',
                                          '/home/bits/arcface-tf2/data/test_dataset/kface1/kface1/kfcae.npy'))
