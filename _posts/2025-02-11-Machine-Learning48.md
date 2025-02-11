---
title: "특강: 인공신경망과 딥러닝 학습 과정 정리"
author: mminzy22
date: 2025-02-11 20:00:00 +0900
categories: [Machine Learning]
tags: [Bootcamp, Python, Machine Learning, Deep Learning, TIL]
description: "퍼셉트론과 다층 퍼셉트론(MLP)의 개념, 신경망의 학습 과정, 경사하강법, 활성화 함수 등을 설명하고, PyTorch를 활용한 신경망 구현 예제를 제공합니다."
pin: false
math: true
---


## 1. 퍼셉트론(Perceptron)과 다층 퍼셉트론(MLP)
### 1.1 퍼셉트론의 동작 원리
퍼셉트론은 가장 기본적인 인공 신경망 모델로, 다음과 같은 단계를 거쳐 동작합니다.

1. 입력값을 받아온다.
2. 각 입력값에 가중치(weight)를 곱한다.
3. 바이어스(bias)를 더해 가중합을 구한다.
4. 활성화 함수(Activation Function)를 적용하여 최종 출력을 결정한다.

### 1.2 퍼셉트론의 한계와 MLP의 등장
퍼셉트론은 **선형 분리 문제**(AND, OR 연산)만 해결할 수 있지만, **XOR 문제**와 같은 비선형 문제는 해결할 수 없습니다. 이를 극복하기 위해 **다층 퍼셉트론(MLP, Multi-Layer Perceptron)** 이 등장했습니다.

MLP는 여러 개의 퍼셉트론을 엮어서 **입력층(Input Layer), 은닉층(Hidden Layer), 출력층(Output Layer)** 으로 구성되며, 은닉층을 추가함으로써 **비선형 문제도 해결**할 수 있습니다.

## 2. 신경망의 학습 과정

### 2.1 순전파(Forward Propagation)
입력 데이터를 신경망에 전달하여 예측값을 계산하는 과정입니다.

1. 입력값을 첫 번째 층에 전달
2. 각 층에서 가중치와 활성화 함수를 적용하여 출력 계산
3. 마지막 출력층에서 최종 예측값 도출

### 2.2 손실 계산(Loss Calculation)
신경망이 예측한 값과 실제 정답 값의 차이를 계산합니다. **손실 함수(Loss Function)** 를 사용하여 오차를 정량화하며, 대표적인 손실 함수로는 **MSE(Mean Squared Error)** 등이 있습니다.

### 2.3 역전파(Backpropagation)
손실을 최소화하기 위해 가중치를 조정하는 과정으로, **기울기(Gradient)를 계산하여 역으로 전파**하는 방식입니다.

1. 순전파를 통해 예측값을 계산
2. 손실 함수로 실제값과 예측값의 차이를 계산
3. 오차를 각 가중치에 대해 미분하여 기울기 계산
4. 기울기를 반영하여 가중치를 업데이트 (경사하강법 적용)

## 3. 경사하강법(Gradient Descent)과 최적화 방법
경사하강법은 손실 함수의 최솟값을 찾기 위해 사용되는 최적화 알고리즘입니다.

### 3.1 경사하강법의 동작 원리
1. 임의의 가중치를 설정
2. 현재 가중치의 **기울기(Gradient)를 계산**
3. 기울기의 반대 방향으로 가중치를 조금씩 조정 (학습률 $$ \alpha $$ 을 곱해서 조절)
4. 위 과정을 반복하여 최적의 가중치를 찾음

### 3.2 경사하강법의 한계
- 지역 최소값(Local Minimum)에 빠질 가능성이 있음
- 기울기 소실(Vanishing Gradient) 문제 발생 가능

이를 해결하기 위해 **확률적 경사하강법(SGD, Stochastic Gradient Descent)** 이나 다양한 최적화 알고리즘(Adam, RMSprop 등)이 사용됩니다.

## 4. 활성화 함수(Activation Function)
활성화 함수는 신경망이 **비선형성을 학습**할 수 있도록 도와줍니다.

### 4.1 주요 활성화 함수
1. **시그모이드(Sigmoid):** 0~1 사이의 값을 출력하며, 주로 확률 계산에 사용됨.
2. **ReLU(Rectified Linear Unit):** 음수 입력을 0으로 변환하여 기울기 소실 문제를 일부 해결.
3. **소프트맥스(Softmax):** 여러 클래스 중 하나를 선택하는 다중 분류 문제에 사용됨.

## 5. 신경망 구현 (PyTorch 활용)
다층 퍼셉트론(MLP)을 PyTorch로 구현하는 예제입니다.

```python
import torch
import torch.nn as nn

# 다층 퍼셉트론(MLP) 모델 정의
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)  # 은닉층
        self.relu = nn.ReLU()  # 활성화 함수
        self.output = nn.Linear(hidden_size, output_size)  # 출력층
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# 모델 생성
model = MLPModel(input_size=4, hidden_size=10, output_size=3)
print(model)
```

## 6. 신경망 학습 과정 정리
1. **순전파(Forward Propagation):** 입력값을 전달하여 예측값을 계산.
2. **손실 계산(Loss Calculation):** 실제값과 예측값의 차이를 손실 함수로 측정.
3. **역전파(Backpropagation):** 기울기를 계산하여 가중치를 업데이트.
4. **경사하강법 적용:** 최적의 가중치를 찾아 손실을 최소화.

## 7. 마무리
**퍼셉트론, 다층 퍼셉트론(MLP), 신경망 학습 과정(순전파, 역전파), 경사하강법, 활성화 함수** 등의 개념을 정리했습니다. 또한 PyTorch를 활용하여 간단한 신경망 모델을 구현하는 방법도 소개했습니다. 딥러닝의 핵심 원리를 이해하면 보다 복잡한 신경망 모델을 구축하고 최적화할 수 있습니다.
