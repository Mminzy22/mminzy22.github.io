---
title: "도미와 빙어 데이터를 활용한 지도 학습 실습"
author: mminzy22
date: 2024-12-09 10:00:00 +0900
categories: [Machine Learning]
tags: [Bootcamp, Python, Machine Learning, TIL]
description: "지도 학습(Supervised Learning)의 대표적인 알고리즘인 K-최근접 이웃(K-Nearest Neighbors, KNN)을 실습하면서 기본 개념과 과정"
pin: false
---


머신러닝은 데이터를 기반으로 패턴을 학습하고 예측하는 기술입니다. 오늘은 **지도 학습(Supervised Learning)**의 대표적인 알고리즘인 **K-최근접 이웃(K-Nearest Neighbors, KNN)**을 실습하면서 기본 개념과 과정을 살펴보겠습니다.


### 🎯 머신러닝 학습 유형 요약

1. **지도 학습 (Supervised Learning)**
   - **정답(타깃)**이 있는 데이터를 이용하여 학습.
   - 새로운 데이터의 정답을 예측하는 데 사용.
   - **특성(feature)**: 데이터를 설명하는 속성. 예를 들어, 생선의 길이와 무게.
   - **대표 알고리즘**: K-최근접 이웃(KNN), 선형 회귀, 의사결정 트리 등.
   - **추가 팁**:
     - 데이터의 **질**이 학습 결과에 큰 영향을 미친다. 훈련 데이터는 가능한 한 다양한 상황을 포함해야 한다.
     - 훈련 세트와 테스트 세트의 분리를 통해 모델의 일반화 능력을 평가할 수 있다.

2. **비지도 학습 (Unsupervised Learning)**
   - **타깃이 없는 데이터**에서 패턴을 찾는 방식.
   - 데이터의 구조를 이해하거나 그룹으로 나누는 데 활용.
   - **대표 알고리즘**: K-평균 군집화, PCA(주성분 분석) 등.
   - **추가 팁**:
     - 데이터의 특성을 제대로 파악하지 못하면 결과 해석이 어려울 수 있다.
     - 시각화 도구를 활용해 데이터의 분포와 군집을 확인하면 유용하다.

3. **강화 학습 (Reinforcement Learning)**
   - **보상**을 통해 학습하며, 특정 작업의 성과를 극대화하는 데 사용.
   - **추가 팁**:
     - 강화 학습은 환경과 상호작용하며 결과를 학습하므로, 시뮬레이션 환경 구축이 중요하다.
     - 게임 AI, 로봇 제어 등에서 주로 활용.


### 1. 데이터 준비 및 시각화

#### 1.1 도미 데이터 준비

```python
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
```

#### 1.2 빙어 데이터 준비

```python
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```

#### 1.3 데이터 시각화

```python
import matplotlib.pyplot as plt

# 도미 데이터
plt.scatter(bream_length, bream_weight, label="Bream") 

# 빙어 데이터
plt.scatter(smelt_length, smelt_weight, label="Smelt") 

plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.legend()  # 범례 추가
plt.title("Scatter Plot of Bream and Smelt")
plt.show()
```


#### 📌 추가 팁: 시각화에 대한 이해
1. **산점도(Scatter Plot):** 
   - 산점도는 데이터를 직관적으로 이해하는 데 유용합니다. 여기서는 생선의 길이와 무게의 관계를 시각화합니다.
   - 도미와 빙어의 데이터를 한눈에 비교할 수 있습니다.

2. **맷플롯립 활용 팁:**
   - `label`: 데이터를 명확히 구분하기 위해 범례를 추가합니다.
   - `title`: 그래프 제목을 추가하면 목적을 명확히 전달할 수 있습니다.

3. **데이터 분포 확인:**
   - 도미는 길이와 무게의 상관관계가 높아 **선형적 분포**를 보입니다.
   - 빙어는 길이와 무게의 변화가 덜하며 상대적으로 **완만한 증가**를 나타냅니다.
![결과 산점도]({{ site.baseurl }}/assets/images/2024-12-09_산점도01.png)


### 2. 데이터 준비와 모델 학습

#### 2.1 데이터 통합 및 타깃 데이터 생성

```python
# 생선 데이터 통합
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# zip()을 이용한 2차원 리스트 생성
fish_data = [[l, w] for l, w in zip(length, weight)]

# 타깃 데이터 생성: 도미는 1, 빙어는 0
fish_target = [1] * 35 + [0] * 14
```


#### 2.2 데이터 준비에 대한 팁

1. **zip 함수의 활용:**
   - `zip` 함수는 여러 리스트를 병렬로 처리할 때 유용합니다. 생선의 길이와 무게 데이터를 각각 묶어 2차원 리스트로 변환합니다.
   - 예: `zip([1, 2], [3, 4])` → `[(1, 3), (2, 4)]`.

2. **타깃 데이터의 의미:**
   - `fish_target` 리스트에서 1은 도미, 0은 빙어를 나타냅니다.
   - 이는 머신러닝 알고리즘이 도미(1)와 빙어(0)를 구분할 수 있도록 훈련시키기 위한 **정답 데이터(라벨)**입니다.


#### 2.3 모델 생성 및 훈련

```python
# 사이킷런의 KNeighborsClassifier 클래스 임포트
from sklearn.neighbors import KNeighborsClassifier

# 모델 객체 생성
kn = KNeighborsClassifier()

# 모델 훈련
kn.fit(fish_data, fish_target)
```


#### 2.4 학습 단계에 대한 팁

1. **모델 생성:**
   - `KNeighborsClassifier()`는 k-최근접 이웃 알고리즘(KNN)을 구현한 클래스입니다.
   - 기본값으로 이웃 수(`n_neighbors`)는 5로 설정되어 있으며, 이를 필요에 따라 조정할 수 있습니다.

2. **모델 훈련:**
   - `fit(fish_data, fish_target)` 메서드는 데이터를 기반으로 알고리즘을 학습시킵니다.
   - KNN은 데이터를 저장해 두었다가, 새로운 데이터의 가까운 이웃을 찾아 예측에 활용합니다.

3. **KNN 알고리즘의 특징:**
   - 메모리에 데이터를 저장하므로 **훈련 과정은 간단**하지만, 예측 시 거리 계산이 필요하므로 데이터가 많을수록 **속도가 느려질 수 있음**.


#### 2.5 학습된 모델 평가

```python
# 모델 평가
score = kn.score(fish_data, fish_target)
print(f"Model Accuracy: {score:.2f}")  # 정확도 출력
```


#### 📌 추가 팁: 모델 평가

1. **`score()` 메서드:**
   - `score()`는 모델의 정확도를 계산합니다. 반환 값은 0~1 사이로, 1에 가까울수록 예측 성능이 좋음을 의미합니다.
   - 정확도(Accuracy) 계산 공식:  
     \\[
     \text{Accuracy} = \frac{\text{정확히 예측한 샘플 수}}{\text{전체 샘플 수}}
     \\]

2. **평가 해석:**
   - 예를 들어, 정확도가 1.0이라면 모든 데이터를 완벽히 예측했다는 뜻입니다. 그러나 이는 데이터 과적합(overfitting) 문제일 수도 있으므로 주의해야 합니다.


### 3. 모델 평가 및 예측

#### 3.1 모델의 정확도 평가

```python
# 학습된 모델의 정확도 평가
model_accuracy = kn.score(fish_data, fish_target)
print(f"Model Accuracy: {model_accuracy:.2f}")  # 정확도 출력
```


#### 📌 모델 평가에 대한 팁

1. **정확도란?**
   - 정확도(Accuracy)는 전체 데이터 중 **올바르게 예측된 데이터의 비율**을 나타냅니다.
   - 정확도 계산 공식:
     \\[
     \text{Accuracy} = \frac{\text{정확히 예측한 샘플 수}}{\text{전체 샘플 수}}
     \\]

2. **훈련 데이터로 평가할 때 주의점:**
   - 훈련 데이터로 평가하면 높은 정확도가 나올 수 있습니다.
   - 일반적으로 훈련 데이터가 아닌 **테스트 데이터**로 평가해야 모델의 일반화 성능을 측정할 수 있습니다.


#### 3.2 새로운 데이터 예측

```python
# 새로운 샘플 데이터 예측
sample = [[25, 150]]  # 길이 25cm, 무게 150g인 생선
prediction = kn.predict(sample)

# 예측 결과 출력
if prediction[0] == 1:
    print("Prediction: 도미(Bream)")
else:
    print("Prediction: 빙어(Smelt)")
```


#### 📌 새로운 데이터 예측 과정

1. **`predict()` 메서드:**
   - `predict()`는 새로운 데이터에 대한 **분류 결과를 반환**합니다.
   - 입력값은 **2차원 리스트** 형태로 전달해야 합니다.
   - 예: `kn.predict([[길이, 무게]])`

2. **예측 결과 해석:**
   - KNN 알고리즘은 가장 가까운 k개의 이웃 중 다수가 속한 클래스를 예측 값으로 반환합니다.
   - 위 예시에서 길이 25cm, 무게 150g인 생선이 **빙어(Smelt)**로 예측될 수 있습니다.


#### 3.3 도미와 빙어 분류 한계

```python
# 새로운 샘플 데이터의 특징이 도미 데이터와 비슷하지만 예측 결과가 빙어일 수 있음
sample_result = kn.predict([[30, 600]])  # 길이 30cm, 무게 600g
print("Prediction:", "도미" if sample_result[0] == 1 else "빙어")
```


#### 📌 추가 팁: 데이터와 모델의 한계 이해

1. **특성 간 중요도 차이:**
   - 예측이 도미 데이터와 비슷한 특징을 가지고 있음에도 빙어로 분류된다면, 이는 **특성 스케일이 달라서** 발생할 수 있습니다.
   - 길이(25cm)와 무게(150g)의 크기 차이가 클 경우, KNN은 거리 계산에서 **무게가 길이에 비해 더 큰 영향을 끼칠 수 있음**.

2. **KNN 알고리즘의 특징:**
   - 모든 특성이 **동등하게 고려**되므로, **특성 간 스케일 차이**가 예측에 영향을 미칩니다.
   - 이러한 문제는 **특성 스케일 조정**(예: 표준화)으로 해결할 수 있습니다.

3. **모델 일반화:**
   - 훈련 데이터 외에 새로운 데이터를 잘 예측하려면, 다양한 상황에서 훈련 데이터와 테스트 데이터를 골고루 섞어 사용하는 것이 중요합니다.


### 4. 핵심 패키지와 함수

#### 4.1 Python 핵심 패키지

1. **NumPy**  
   - **`numpy`**는 **수학 연산 및 배열 연산**을 지원하는 Python 라이브러리입니다.
   - **랜덤 샘플링 및 데이터 처리**에도 유용합니다.

   ```python
   import numpy as np
   ```

   - 주요 함수 및 메서드:
     - **`np.array()`**: 파이썬 리스트를 넘파이 배열로 변환.
     - **`np.mean()` / `np.std()`**: 평균 및 표준편차 계산.
     - **`np.arange()`**: 지정된 범위의 숫자로 배열 생성.
     - **`np.random.shuffle()`**: 배열 섞기.
     - **`np.column_stack()`**: 여러 배열을 열 방향으로 병합.

   ```python
   # 예시
   fish_data = np.column_stack((fish_length, fish_weight))  # 데이터 병합
   print(fish_data[:5])
   ```


2. **Matplotlib**  
   - **`matplotlib.pyplot`**는 **데이터 시각화**를 위한 도구입니다.
   - 산점도, 선 그래프 등 다양한 그래프를 그릴 수 있습니다.

   ```python
   import matplotlib.pyplot as plt
   ```

   - 주요 함수 및 메서드:
     - **`plt.scatter(x, y)`**: 산점도를 그립니다.
     - **`plt.xlabel()` / `plt.ylabel()`**: 축 레이블 추가.
     - **`plt.show()`**: 플롯(그래프) 표시.

   ```python
   # 예시
   plt.scatter(bream_length, bream_weight)
   plt.xlabel("Length")
   plt.ylabel("Weight")
   plt.show()
   ```


#### 4.2 Scikit-learn 핵심 함수 및 클래스

1. **KNeighborsClassifier**
   - k-최근접 이웃(KNN) 알고리즘을 구현한 클래스입니다.
   - 이웃 개수를 설정하고 데이터를 학습, 예측합니다.

   ```python
   from sklearn.neighbors import KNeighborsClassifier
   ```

   - 주요 메서드:
     - **`fit(X, y)`**: 모델 학습.
     - **`predict(X)`**: 예측값 반환.
     - **`score(X, y)`**: 정확도 계산.

   ```python
   # 예시
   kn = KNeighborsClassifier(n_neighbors=5)  # k=5 설정
   kn.fit(train_input, train_target)  # 모델 학습
   accuracy = kn.score(test_input, test_target)  # 정확도
   print(f"Accuracy: {accuracy:.2f}")
   ```


2. **train_test_split**
   - 데이터를 훈련 세트와 테스트 세트로 분리합니다.

   ```python
   from sklearn.model_selection import train_test_split
   ```

   - 주요 매개변수:
     - **`test_size`**: 테스트 세트 비율 설정(기본값 0.25).
     - **`random_state`**: 랜덤 시드 고정.
     - **`stratify`**: 클래스 비율을 유지하도록 데이터 나눔.

   ```python
   # 예시
   train_input, test_input, train_target, test_target = train_test_split(
       fish_data, fish_target, test_size=0.2, stratify=fish_target, random_state=42
   )
   ```


3. **kneighbors**
   - 주어진 데이터에 대해 가장 가까운 이웃을 찾습니다.

   ```python
   distances, indexes = kn.kneighbors([[30, 600]])
   ```

   - 반환값:
     - **`distances`**: 이웃과의 거리.
     - **`indexes`**: 이웃의 인덱스.

   ```python
   # 예시
   print("Distances:", distances)
   print("Indexes:", indexes)
   ```


#### 📌 팁: 학습 유형별 모델 사용

- **지도 학습**: KNeighborsClassifier와 같은 분류 알고리즘.
- **비지도 학습**: 데이터 클러스터링이나 차원 축소.
- **강화 학습**: 보상 기반의 학습으로 별도의 환경 시뮬레이터가 필요.


### 5. 정리

이번 학습에서는 **지도 학습**을 중심으로 데이터 준비, 모델 학습, 평가, 예측의 전체 과정을 경험했습니다. 아래는 주요 학습 내용과 얻은 통찰을 정리한 내용입니다.


#### 1. 데이터 준비와 전처리의 중요성
- **데이터 준비**는 머신러닝의 첫 단계입니다. 
  - 데이터를 잘 준비하고 가공하면 학습 성능을 극대화할 수 있습니다.
  - 특성을 정의하고, 데이터를 적절히 나누는 작업이 필요합니다.
- **넘파이**를 활용해 데이터를 배열로 변환하고 병합하는 등의 작업이 이루어졌습니다.
  - 이를 통해 머신러닝 모델이 이해할 수 있는 형태로 데이터를 변환했습니다.


#### 2. 지도 학습의 기본 원리
- **지도 학습**은 **입력(특성)**과 **출력(타깃)**을 학습하여 새로운 데이터를 예측하는 방식입니다.
  - 이번 실습에서는 **생선의 길이와 무게**를 사용하여 도미와 빙어를 분류했습니다.
- **훈련 데이터와 테스트 데이터의 구분**은 모델 성능 평가에서 매우 중요합니다.
  - 훈련 데이터로 학습하고 테스트 데이터로 평가함으로써 모델의 일반화 능력을 검증했습니다.


#### 3. K-최근접 이웃 알고리즘(KNN)의 특징
- **KNN 알고리즘**은 단순하면서도 강력한 분류 알고리즘입니다.
  - 데이터 간의 거리를 계산하여 가장 가까운 이웃의 다수결로 분류합니다.
- 모델 훈련 과정에서 실제로 규칙을 생성하지 않고 데이터를 저장합니다.
  - **n_neighbors** 매개변수는 모델 성능에 큰 영향을 미칩니다.
  - 적절한 이웃 개수를 선택하는 것이 중요합니다.


#### 4. 평가와 정확도
- **score() 메서드**를 사용하여 모델의 정확도를 확인했습니다.
  - 높은 정확도는 모델이 주어진 데이터에서 잘 작동함을 의미합니다.
- 예측 결과를 **산점도**로 시각화하여 모델의 성능을 직관적으로 확인했습니다.


#### 5. 데이터 전처리와 표준화
- 두 특성의 스케일이 다르면 거리 기반 알고리즘에서 문제가 발생할 수 있습니다.
  - **표준화**를 통해 데이터의 평균과 표준편차를 맞추는 작업이 이루어졌습니다.
  - 표준화된 데이터는 학습과 예측 성능을 향상시킵니다.


#### 6. 학습 유형별 적용 사례
- **지도 학습**: KNN을 사용하여 도미와 빙어를 분류.
- **비지도 학습**: 타깃 데이터 없이 클러스터링 및 데이터 탐색.
- **강화 학습**: 행동의 결과로 얻은 보상을 바탕으로 학습.


#### 🌟 배운 점
1. **데이터 준비와 가공은 모델 학습의 성패를 좌우한다.**
2. **KNN 알고리즘은 데이터의 분포를 이해하는 데 유용하지만, 데이터 양이 많아지면 효율성이 떨어질 수 있다.**
3. **거리 기반 알고리즘을 사용할 때는 특성 간 스케일을 조정하는 것이 중요하다.**
4. **훈련 세트와 테스트 세트는 반드시 분리하여 모델의 일반화 능력을 평가해야 한다.**
