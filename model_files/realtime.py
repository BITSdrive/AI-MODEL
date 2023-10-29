from deepface import DeepFace
from commons import functions, distance as dst
import cv2
import numpy as np
import ArcFace
import firebase_admin
from firebase_admin import credentials, messaging
import time

# Firebase 초기화
cred = credentials.Certificate("C:\\Users\\HanGyeol.Kim\\Desktop\\asd\\bits-drive-bcab9-firebase-adminsdk-ibjho-5235e7f2f6.json")
firebase_admin.initialize_app(cred)

# 모델 로딩
model = ArcFace.loadModel()
model.load_weights("arcface_weights.h5")

target_size = (112, 112)
img2_path = "C:\\Users\\HanGyeol.Kim\\Desktop\\asd\\fa\\hg1.jpg"
detector_backend = 'retinaface'

# 기준 이미지 로딩
img2 = functions.extract_faces(img2_path, target_size=target_size, detector_backend=detector_backend)

metric = 'cosine'


def findThreshold(metric):
    if metric == 'cosine':
        return 0.6871912959056619
    elif metric == 'euclidean':
        return 4.1591468986978075
    elif metric == 'euclidean_l2':
        return 1.1315718048269017


def send_notification_to_android():
    # Firebase를 사용하여 Android 앱에 푸시 알림을 전송
    message = messaging.Message(
        data={'alert': 'Different person detected multiple times!'},
        topic='your_android_app_topic'
    )
    messaging.send(message)


def verify(img1, img2):
    img1_array = np.array([face_img for face_img, _, _ in img1])
    img2_array = np.array([face_img for face_img, _, _ in img2])

    # 얼굴이 감지되지 않았을 경우 카운트 및 경고를 하지 않음
    if len(img1_array) == 0 or len(img2_array) == 0:
        return

    img1_array = np.squeeze(img1_array, axis=1)
    img2_array = np.squeeze(img2_array, axis=1)
    img1_embedding = model.predict(img1_array)[0]
    img2_embedding = model.predict(img2_array)[0]

    if metric == 'cosine':
        distance = dst.findCosineDistance(img1_embedding, img2_embedding)
    elif metric == 'euclidean':
        distance = dst.findEuclideanDistance(img1_embedding, img2_embedding)
    elif metric == 'euclidean_l2':
        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_embedding), dst.l2_normalize(img2_embedding))

    threshold = findThreshold(metric)
    if distance > threshold:
        return "different"
    else:
        return "same"



def save_notification_to_local():
    alert_message = 'Different person detected multiple times!'
    with open('alert_message.txt', 'w') as txt_file:
        txt_file.write(alert_message)

# 웹캠 초기화
cap = cv2.VideoCapture(0)
diff_count = 0  # 다른 사람으로 판별된 횟수를 저장하는 변수

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    try:
        # 임시 파일 저장 및 로드 없이 직접 프레임에서 얼굴 추출
        img1 = functions.extract_faces(frame, target_size=target_size, detector_backend=detector_backend)

        # img1에서 얼굴이 감지되면 verify를 수행
        if img1:
            result = verify(img1, img2)
            if result == "different":
                diff_count += 1

    except ValueError:
        pass  # 얼굴 감지 실패시 경고메시지 무시하고 계속 실행

    if diff_count >= 5:
        send_notification_to_android()
        save_notification_to_local()
        diff_count = 0

    time.sleep(5)

cap.release()
cv2.destroyAllWindows()