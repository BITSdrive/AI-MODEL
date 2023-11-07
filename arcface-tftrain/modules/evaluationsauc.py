import os
import cv2
import bcolz
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tqdm
from modules.utils import l2_norm

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))
    return carray, issame

def get_val_data(data_path):
    lfw, lfw_issame = get_val_pair(data_path, 'kface1/kface1')
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_align_112/agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_align_112/cfp_fp')
    return lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame

def ccrop_batch(imgs):
    assert len(imgs.shape) == 4
    resized_imgs = np.array([cv2.resize(img, (128, 128)) for img in imgs])
    ccropped_imgs = resized_imgs[:, 8:-8, 8:-8, :]
    return ccropped_imgs

def hflip_batch(imgs):
    assert len(imgs.shape) == 4
    return imgs[:, :, ::-1, :]

def evaluate(embeddings, actual_issame):
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    tpr, fpr, thresholds = roc_curve(np.asarray(actual_issame), dist)
    roc_auc = auc(1-tpr, fpr)  # Adjusted the AUC calculation 

    # Calculate fnir and adjust the AUC score
    fnir = 1 - tpr

    # Calculate accuracy and best_thresholds
    accuracy = (tpr + (1 - fpr)) / 2
    best_idx = np.argmax(accuracy)
    best_thresholds = thresholds[best_idx]

    # Plot and save ROC curve
    save_dir = "saved_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure()
    plt.plot(fnir, fpr, color='darkorange', label='Curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [1, 0], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Negative Identification Rate (FNIR)')
    plt.ylabel('False Positive Identification Rate (FPIR)')
    plt.title('FNIR vs FPIR Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'fn_vs_fp_curve.png'))
    plt.show()

    with open(os.path.join(save_dir, 'auc_value.txt'), 'w') as f:
        f.write('AUC: %0.2f' % roc_auc) 

    return tpr, fpr, accuracy[best_idx], best_thresholds 

def perform_val(embedding_size, batch_size, model, carray, issame, is_ccrop=False, is_flip=True):
    embeddings = np.zeros([len(carray), embedding_size])
    for idx in tqdm.tqdm(range(0, len(carray), batch_size)):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1]) * 0.5 + 0.5
        batch = batch[:, :, :, ::-1]
        if is_ccrop:
            batch = ccrop_batch(batch)
        if is_flip:
            fliped = hflip_batch(batch)
            emb_batch = model(batch) + model(fliped)
            embeddings[idx:idx + batch_size] = emb_batch
        else:
            emb_batch = model(batch)
            embeddings[idx:idx + batch_size] = emb_batch
            
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame)
    return accuracy, best_thresholds
    
    
    
def visualize_sample(carray, num_samples=5):
    """
    데이터셋에서 무작위로 샘플 이미지를 시각화합니다.
    """
    indices = np.random.choice(len(carray), num_samples)
    for i in indices:
        image = carray[i].transpose(1, 2, 0) * 0.5 + 0.5  # [-1, 1] 범위에서 [0, 1] 범위로 복원
        plt.imshow(image)
        plt.title(f"Index: {i}")
        plt.show()

def dataset_statistics(carray):
    """
    데이터셋의 기본 통계를 반환합니다.
    """
    mean = np.mean(carray, axis=(0, 2, 3))
    std = np.std(carray, axis=(0, 2, 3))
    return mean, std

def check_class_distribution(issame):
    """
    클래스 분포를 확인하고 출력합니다.
    """
    unique, counts = np.unique(issame, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} samples")

if __name__ == "__main__":
    data_path = "/your/test/dataset/path"
    lfw, _, _, lfw_issame, _, _ = get_val_data(data_path)

    # 데이터셋의 일부를 시각화합니다.
    visualize_sample(lfw)

    # 데이터셋의 통계를 계산하고 출력합니다.
    mean, std = dataset_statistics(lfw)
    print("Mean:", mean)
    print("Standard Deviation:", std)
    
    # 클래스 분포를 확인합니다.
    check_class_distribution(lfw_issame)    

