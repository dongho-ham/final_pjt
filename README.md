# Battery Anomaly Detection using Deep Learning and Machine Learning

NASA 배터리 데이터셋을 활용한 시계열 이상 탐지 프로젝트

## 프로젝트 개요

이차전지 충방전 사이클 데이터에서 **Anomaly Transformer** 모델을 사용하여 배터리 이상 패턴을 탐지합니다.

## 실행 방법

전처리
python preprocessing/build_dataset

학습
cd Anomaly-Transformer
python main.py --mode train

## 기술 스택

- PyTorch
- Anomaly Transformer (ICLR 2022)
- LOWESS Smoothing