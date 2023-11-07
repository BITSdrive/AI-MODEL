import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from absl import app, flags, logging
from trainnn import get_val_data, perform_val, visualize_confusion_matrix
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm
from absl.flags import FLAGS

flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('img_path', '', 'path to input image')

def visualize_sample(carray, num_samples=5):
    """
    데이터셋에서 무작위로 샘플 이미지를 시각화합니다.
    """
    indices = np.random.choice(len(carray), num_samples)
    for i in indices:
        image = carray[i].transpose(1, 2, 0) * 0.5 + 0.5  # [-1, 1] 범위에서 [0, 1] 범위로 복원
        plt.imshow(image)
        
        min_val = image.min()
        max_val = image.max()
        mean_val = image.mean()
        title_text = f"Index: {i}\nMin: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}"
        plt.title(title_text)
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

# MirroredStrategy 초기화
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # ArcFaceModel 및 체크포인트 로드
    with strategy.scope():
        model = ArcFaceModel(size=cfg['input_size'],
                             backbone_type=cfg['backbone_type'],
                             training=False)
        ckpt_path = tf.train.latest_checkpoint('/home/bits/arcface-tf2/checkpoints/arc_res50pre/')

        if ckpt_path is not None:
            print("[*] load ckpt from {}".format(ckpt_path))
            model.load_weights(ckpt_path)
        else:
            print("[*] Cannot find ckpt from {}.".format(ckpt_path))
            exit()

    if FLAGS.img_path:
        print("[*] Encode {} to ./output_embeds.npy".format(FLAGS.img_path))
        img = cv2.imread(FLAGS.img_path)
        img = cv2.resize(img, (cfg['input_size'], cfg['input_size']))
        img = img.astype(np.float32) / 255.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        embeds = l2_norm(model(img))
        np.save('./output_embeds.npy', embeds)
    else:
        print("[*] Loading LFW, AgeDB30 and CFP-FP...")
        lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
            get_val_data(cfg['test_dataset'])
        visualize_sample(lfw)
        visualize_sample(agedb_30)
        mean, std = dataset_statistics(lfw)
        mean, std = dataset_statistics(agedb_30)
        print("Mean of the dataset:", mean)
        print("Standard Deviation of the dataset:", std)
        
        print("[*] Perform Evaluation on LFW...")
        acc_lfw, best_th = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, lfw, lfw_issame,
            is_ccrop=cfg['is_ccrop'])
        print("    acc {:.4f}, th: {:.2f}".format(acc_lfw, best_th))

        print("[*] Perform Evaluation on AgeDB30...")
        acc_agedb30, best_th = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, agedb_30,
            agedb_30_issame, is_ccrop=cfg['is_ccrop'])
        print("    acc {:.4f}, th: {:.2f}".format(acc_agedb30, best_th))

        print("[*] Perform Evaluation on CFP-FP...")
        acc_cfp_fp, best_th = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, cfp_fp, cfp_fp_issame,
            is_ccrop=cfg['is_ccrop'])
        print("    acc {:.4f}, th: {:.2f}".format(acc_cfp_fp, best_th))
        check_class_distribution(lfw_issame)
        check_class_distribution(agedb_30_issame)
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

