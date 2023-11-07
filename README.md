# AI-MODEL : ArcFace
- **ArcFace** 
>딥러닝 ArcFace 모델의 추론 결과 반환은 얼굴 인식 및 얼굴 임베딩 분야에서 중요한 부분을 차지하고 있습니다. 이러한 모델은 주로 얼굴 이미지를 입력으로 받아, 이를 임베딩 벡터로 변환하고 얼굴 간의 유사성을 측정함으로써 얼굴 인식 작업을 수행합니다. 추론 결과를 반환하는 이러한 모델은 다양한 응용 분야에서 활용되며, 보안 시스템, 얼굴 감지, 엑세스 제어, 자동화된 인증, 현실 세계에서의 얼굴 식별 등 다양한 용도로 활용됩니다.

## **Model Architecture**
![모델 구조](https://github.com/BITSdrive/AI-MODEL/assets/126750984/4cc09397-6911-4a6f-ada5-0db7d07422d7)

 - 특징 <br/>
    &nbsp;ArcFace 는 소프트맥스 손실 함수에 각도 마진을 추가하여, 클래스 간의 결정 경계를 더 명확하게 하고, 
    동시에 클래스 내의 특징을 더 긴밀하게 모으는 효과를 제공합니다. 이는 기존의 소프트맥스 손실 함수가 클래스 간 구별력을 최적화하는 데 한계가 있다는 점을 개선하여, 
    특히 얼굴 인식과 같은 정밀한 식별 작업에서 더 우수한 성능을 나타냅니다.
<br/>
<br/>

## 📖 참고자료

- ArcFace 논문 &nbsp; : &nbsp; ArcFace: Additive Angular Margin Loss for Deep Face Recognition https://arxiv.org/abs/1801.07698  (2022)

- insightface &nbsp; : &nbsp;  https://github.com/deepinsight/insightface/tree/master/recognition (Official)

<br/>
<br/>

## Model Train Environment

- ubuntu 22.04 버전
- Tensorflow
- Tensorflow RT
- bcolz



## Train Dataset
- MS-Celeb-1M  &nbsp; (pretrained data)
> &nbsp; Microsoft Research에서 얼굴 인식 연구를 위해 공개한 대규모 데이터 셋입니다.
이 데이터셋은 약 100,000명의 유명 셀럽에 대한 1백만개 이상의 얼굴 이미지를 포함하고 있으며, 다양성과 대규모 데이터 셋이라는 
장점으로 얼굴 인식 모델의 학습 데이터로 널리 사용되고있습니다.

<br/>

- KoreanFace &nbsp;  (fine-tuning data)
> &nbsp; 한국인 얼굴 사진 데이터베이스는 디지털 뉴딜 사업의 일환으로 인공지능 학습용 데이터 집합소인 AI hub를 통해 구축하였습니다.
Korean face는 총 1300명의 라벨과 각 인물당 50장의 이미지를 사용하였으며, (1300x50 = 65,000)
원본데이터를 Arcface의 입력 사이즈 형태로 가공하기위해 RetinaFace 라이브러리를 사용하여 얼굴을 (112,112) 사이즈로 정렬하는 과정을 거쳤습니다.
유의할 점으로 데이터는 AI Hub의 사전 허가를 통해 제공받을 수 있으며 허가 받지 않은 사용자에게 데이터를 배포를 금하고 있습니다.


<p align="center">
<img width="300" alt="image" src="https://github.com/BITSdrive/AI-MODEL/assets/126750984/a71ce01d-73dd-4b5b-bb3a-60de17b8412f">
</p>

<br/>

## Test Dataset
- test data set으로는 LFW, Aged30, CFP-FP 데이터셋을 사용하였습니다
각 데이터 셋들은 다양한 연령대, 포즈, 조명 조건을 지니고 있기에 얼굴 인증 모델의 테스트 데이터셋으로 널리 사용되고 있는 특징이 있습니다.

<br/>

##  Training and Testing

<p align="center">
<img width="300" alt="image" src="https://github.com/BITSdrive/AI-MODEL/assets/126750984/edf8490a-baca-43de-8c01-826f7008aa39">
</p>
<br/>

- **TFRecord**
> TF프레임 워크에서 대용량의 이미지로 모델을 훈련하기 위해서는 TFrecorder로 이미지 데이터를 직렬화 하는 방식으로 진행하였습니다
TFrecord는 tensorflow에서 사용되는 데이터 형식으로, 대용량의 데이터셋을 다룰 때 사용됩니다
이는 원본 데이터를 직렬화하여 이진 바이너리 파일형태로 저장하기에 Tensorflow의 입출력 작업에 최적화 되어 있어 데이터 로딩시간을 줄여줍니다
face verification을 위한 모델의 학습에는 대용량의 이미지 데이터가 필요하기에 Tensorflow 환경에서 효과적으로 학습하기 위해서 tf-record를 사용하였습니다


<br/>

## Hyperparameter
<p align="center">
<img width="300" alt="image" src="https://github.com/BITSdrive/AI-MODEL/assets/126750984/d9616b48-aa79-4601-af18-40ce4e40c68a">
</p>

- MS-Celeb-1M config &nbsp;
>batch_size: 256 &nbsp;
input_size: 112 &nbsp;
embd_shape: 512, &nbsp;
backbone_type: 'ResNet50', &nbsp;
head_type: ArcHead,, &nbsp;

> train : train_dataset: './data/ms1m_bin.tfrecord', &nbsp;
binary_img: True , &nbsp;
num_classes: 85742 , &nbsp;
num_samples: 5822653, &nbsp;
epochs: 20, &nbsp;
base_lr: 0.01, &nbsp;
w_decay: !!float 5e-4, &nbsp;
save_steps: 1000 &nbsp;

<br/>
<br/>
<br/>

<p align="center">
<img width="200" alt="image" src="https://github.com/BITSdrive/AI-MODEL/assets/126750984/c07f4f1d-1e11-4d4f-9312-f357d1e9d884"> &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img width="200" alt="image" src="https://github.com/BITSdrive/AI-MODEL/assets/126750984/46dd4e66-73e2-4b89-a2e5-db313e3a3e6e"> 
</p>

- K-Face
> MS-Celeb-1M데이터를 사용하여 얻은 가중치를 사용하여 한국인 얼굴 이미지로 파인 튜닝 

- KoreanFace
> config
batch_size: 512 , &nbsp;
input_size: 112, &nbsp;
embd_shape: 512, &nbsp;
sub_name: 'arc_res50', &nbsp;
backbone_type: 'ResNet50, &nbsp;
head_type: ArcHead 

- train
> train_dataset: './data/Koreanfaceparsing.tfrecord'
binary_img: True
num_classes: 1300
num_samples: 65000
epochs: 2000
base_lr: 0.001
w_decay: !!float 5e-4
save_steps: 1000

<br/>
<br/>
<br/>

## **Fine-Tuning**
<img src="https://github.com/BITSdrive/AI-MODEL/assets/126750984/921df2b6-5029-4b85-9da1-430721e44e24" alt="훈련 출력" width = "350"/> <img src="https://github.com/BITSdrive/AI-MODEL/assets/126750984/ad3dd83b-4bf5-4234-88c4-83a8be420297" alt="훈련 출력" width = "360"/>

<br/>
<br/>
<br/>

## Test 지표
- FPIR (False Rejection Rate)
> 실제로는 같은 사람인 경우 중, 알고리즘이 다른 사람이라고 판단한 비율
거짓 수락율

<br/>

- FNIR (False Acceptance Rate)
> 실제로는 다른 사람인 경우 중, 알고리즘이 같은 사람이라고 판단한 비율
> 
<br/>

- 특성곡선 (Receiver Operating Characteristic curve)
> 임계값을 0에서 1까지 변화시키며 각각의 임계값마다 (FAR, FRR) 혹은 (FPIR, FNIR)을 계산해 2차원 좌표에 점들을 찍어 그린 곡선
> 
<br/>

- AUC (Area Under Curve)
> 특성 곡선의 아래 면적을 의미하며,
AUC는 여러 임계값 마다 측정된 오류율을 종합적인 값으로 산출
※ 0에 가까울수록 오류율이 낮다고 해석

<br/>

<p align="center">
<img width="600" alt="image" src="https://github.com/BITSdrive/AI-MODEL/assets/126750984/e96cb13b-acf3-426e-be80-266fccbb16d9">
</p>

<br/>

<br/>
<br/>

## Test Result
- **FPIR , FNIR 곡선**
  
| <img src="https://github.com/BITSdrive/AI-MODEL/assets/126750984/10ee8a3f-b1fc-4652-8f77-1ce4c5acae8c" alt="curve" width = "500"/> | <img src="https://github.com/BITSdrive/AI-MODEL/assets/126750984/69eb35d3-2932-4523-87d3-9033135c0e1f" alt="curve" width = "500"/>  | 
| :---: | :---: | 
|[MS-Celeb-1M] | [K-face fine-tuning]|