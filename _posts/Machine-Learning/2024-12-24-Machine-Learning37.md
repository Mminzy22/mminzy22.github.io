---
layout: post
title: "Machine Learning 37: 과제: 주택 가격 예측 모델 구축(스케일링 및 이상치 처리)"
date: 2024-12-24
categories: [Machine Learning]
tag: []
---

### 과제 1번: 지도학습

#### 주제: 주택 가격 예측 모델 구축
- 주어진 [주택 데이터셋]({{ site.baseurl }}/assets/downloads/housingdata.csv)을 사용하여 주택 가격을 예측하는 회귀 모델을 구축한다.

<details>
<summary><strong>컬럼별 설명</strong></summary>
   <ol>
      <li>CRIM: 타운별 1인당 범죄율.</li>
      <li>ZN: 25,000 평방피트 이상의 주거 구역 비율.</li>
      <li>INDUS: 비소매 상업 지역 비율.</li>
      <li>CHAS: 찰스강 인접 여부 (1: 강과 접함, 0: 접하지 않음).</li>
      <li>NOX: 대기 중 일산화질소 농도 (0.1 단위).</li>
      <li>RM: 주택 1가구당 평균 방 개수.</li>
      <li>AGE: 1940년 이전에 건설된 주택 비율.</li>
      <li>DIS: 5개의 보스턴 고용 중심지까지의 가중 거리.</li>
      <li>RAD: 고속도로 접근성 지수.</li>
      <li>TAX: 10,000달러당 재산세율.</li>
      <li>PTRATIO: 타운별 학생-교사 비율.</li>
      <li>B: 흑인 비율 (1000(Bk - 0.63)^2, 여기서 Bk는 흑인 인구 비율).</li>
      <li>LSTAT: 하위 계층 인구 비율.</li>
      <li>MEDV: 주택의 중앙값 (단위: $1000).</li>
   </ol>
</details>

<details>
<summary><strong>과제 가이드</strong></summary>
   <ul>
      <li>데이터셋 탐색 및 전처리:
         <ul>
            <li>결측치 처리</li>
            <li>이상치 탐지 및 제거</li>
            <li>특징 선택</li>
         </ul>
      </li>
      <li>여러 회귀 모델 비교:
         <ul>
            <li>선형 회귀</li>
            <li>의사결정나무</li>
            <li>랜덤 포레스트 등</li>
         </ul>
      </li>
      <li>모델 성능 평가:
         <ul>
            <li>지표를 사용하여 모델 성능을 비교합니다.
               <ul>
                  <li>Mean Absolute Error (MAE): 예측값과 실제값의 절대 오차의 평균.</li>
                  <li>Mean Squared Error (MSE): 예측값과 실제값의 제곱 오차의 평균.</li>
                  <li>R² Score: 모델이 데이터의 변동성을 얼마나 설명하는지 나타내는 지표.</li>
               </ul>
            </li>
         </ul>
      </li>
      <li>결과 분석:
         <ul>
            <li>각 모델의 성능을 비교하고 최적의 모델을 선택하여 결과를 시각화합니다.
               <ul>
                  <li>시각화: 성능 지표를 막대 그래프로 시각화하여 쉽게 비교할 수 있도록 합니다. matplotlib 또는 seaborn을 사용하여 막대 그래프를 그립니다.</li>
               </ul>
            </li>
         </ul>
      </li>
   </ul>
</details>

<details>
<summary><strong>도전 과제 가이드</strong></summary>
   <ul>
      <li>모델 앙상블 ⭐⭐⭐⭐⭐
         <ul>
            <li>여러 모델의 예측 결과를 결합하여 성능을 향상시키는 앙상블 기법(예: 배깅, 부스팅)을 적용합니다.</li>
            <li>각 모델의 예측을 평균내거나 가중치를 부여하여 최종 예측을 생성합니다.</li>
         </ul>
      </li>
      <li>하이퍼파라미터 튜닝 ⭐⭐⭐⭐
         <ul>
            <li>Grid Search 또는 Random Search 기법을 이용해 모델의 하이퍼파라미터를 최적화합니다.</li>
         </ul>
      </li>
      <li>시간적 요소 추가 ⭐⭐⭐
         <ul>
            <li>주택 데이터셋에 시간적 요소(예: 계절적 변화, 경제 지표 등)를 추가하여 모델의 예측력을 높입니다.</li>
         </ul>
      </li>
   </ul>
</details>


### 1. 데이터 스케일링(표준화)

```python
from sklearn.preprocessing import StandardScaler

# 표준화 (훈련 데이터로부터 학습)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)
```


#### 코드 설명

1. **라이브러리 임포트**:
   - `StandardScaler`: Scikit-learn의 데이터 표준화 클래스입니다. 데이터의 평균을 0으로, 표준 편차를 1로 조정합니다.

2. **스케일링 객체 생성**:
   - `scaler = StandardScaler()`:
     - 표준화를 수행할 `StandardScaler` 객체를 생성합니다.

3. **훈련 데이터 스케일링**:
   - `scaler.fit_transform(X_train_imputed)`:
     - `fit_transform` 메서드를 통해 훈련 데이터를 기준으로 평균과 표준 편차를 계산(`fit`)하고, 데이터를 표준화(`transform`)합니다.
   - `pd.DataFrame(...)`:
     - 표준화된 데이터를 Pandas 데이터프레임으로 변환하고, 원래 열 이름(`X_train.columns`)을 유지합니다.

4. **테스트 데이터 스케일링**:
   - `scaler.transform(X_test_imputed)`:
     - 훈련 데이터에서 계산한 평균과 표준 편차를 기반으로 테스트 데이터를 표준화합니다.
     - 테스트 데이터는 `fit`을 수행하지 않고, 학습된 훈련 데이터의 기준에 맞춰야 데이터 누출(Data Leakage)을 방지할 수 있습니다.


#### 사용 목적
1. **특성 간 크기 차이 조정**:
   - 특성의 값 범위(스케일)가 크게 다를 경우, 머신러닝 모델의 학습 성능에 영향을 줄 수 있습니다. 이를 해결하기 위해 표준화를 사용합니다.
   - 예: 주택 데이터에서 방의 개수(정수)와 면적(제곱미터)의 값 범위가 다를 수 있습니다.

2. **특정 알고리즘의 요구사항**:
   - 거리 기반 알고리즘(예: KNN, SVM) 또는 경사 하강법을 사용하는 알고리즘(예: 선형 회귀, 로지스틱 회귀, 신경망)은 특성 값의 범위에 민감합니다. 이 경우, 스케일링이 필수적입니다.


#### 활용 사례
1. **선형 모델**:
   - 선형 회귀, 로지스틱 회귀, 서포트 벡터 머신(SVM) 등에서 특성의 스케일이 중요합니다.

2. **거리 기반 모델**:
   - KNN, K-평균(K-Means) 등 거리 계산을 사용하는 모델은 스케일링되지 않은 데이터에서 잘못된 결과를 초래할 수 있습니다.

3. **신경망**:
   - 신경망 모델은 데이터 스케일이 적절하지 않으면 학습이 느려지거나 비효율적으로 작동할 수 있습니다.


#### 매개변수 설명
1. **`fit_transform`**:
   - `fit`과 `transform`을 동시에 수행합니다.
   - `fit`: 데이터를 분석하여 평균과 표준 편차를 계산합니다.
   - `transform`: 계산된 평균과 표준 편차를 사용해 데이터를 변환합니다.

2. **`transform`**:
   - 이미 계산된 평균과 표준 편차를 사용하여 데이터를 변환합니다.
   - 테스트 데이터에서는 항상 `transform`만 사용해야 합니다.


#### 사용 시 주의사항
1. **훈련-테스트 데이터 분리**:
   - 훈련 데이터로 계산된 평균과 표준 편차를 테스트 데이터에 사용해야 하며, 테스트 데이터에 대해 `fit_transform`을 수행하면 데이터 누출이 발생합니다.

2. **표준화가 필요한 모델**:
   - 의사결정 나무 기반 모델(예: Random Forest, Gradient Boosting)은 스케일링에 영향을 받지 않으므로 표준화가 불필요합니다.

3. **결측치 처리 후 수행**:
   - 스케일링은 결측치가 없는 상태에서 수행해야 합니다. 그렇지 않으면 오류가 발생하거나 왜곡된 결과가 나올 수 있습니다.


#### 활용 상황
1. **`fit_transform` 사용 (훈련 데이터)**:
   ```python
   X_train_scaled = scaler.fit_transform(X_train_imputed)
   ```
   - 훈련 데이터에서 평균과 표준 편차를 계산하고, 데이터를 변환.

2. **`transform` 사용 (테스트 데이터)**:
   ```python
   X_test_scaled = scaler.transform(X_test_imputed)
   ```
   - 훈련 데이터에서 계산한 평균과 표준 편차를 사용해 테스트 데이터를 변환.

3. **데이터 스케일링이 필요 없는 상황**:
   - 스케일링이 불필요한 모델(예: 결정 트리, 랜덤 포레스트)을 사용하는 경우.


### 2. 이상치 탐지 및 처리

```python
from sklearn.ensemble import IsolationForest, RandomForestRegressor

# 이상치 탐지 및 처리 (훈련 데이터에만 적용)
clf = IsolationForest(contamination=0.05, random_state=42)
X_train_scaled['anomaly'] = clf.fit_predict(X_train_scaled)

# 이상치와 정상 데이터 분리
outliers = X_train_scaled[X_train_scaled['anomaly'] == -1]  # 이상치 행
non_outliers = X_train_scaled[X_train_scaled['anomaly'] != -1]  # 정상 행

# 이상치 대체 (훈련 데이터에서만 처리)
independent_columns = X_train.columns
for col in independent_columns:
    # 현재 열(col)을 제외한 나머지 열(features)에서 이상치 제거
    features = [c for c in independent_columns if c != col]
    clean_features = non_outliers[features]  # 정상 데이터에서만 추출
    target = non_outliers[col]  # 현재 예측 대상 열의 정상 값

    # 모델 학습 (RandomForestRegressor)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(clean_features, target)

    # 이상치 대체 (예측)
    predicted_values = model.predict(outliers[features])
    X_train_scaled.loc[outliers.index, col] = predicted_values

# 'anomaly' 열 제거
X_train_scaled.drop(columns=['anomaly'], inplace=True)
```


#### 코드 설명

1. **라이브러리 임포트**:
   - `IsolationForest`: 이상치 탐지(Anomaly Detection)를 위한 비지도 학습 모델. 데이터 분포에서 이상치를 감지합니다.
   - `RandomForestRegressor`: 이상치를 대체하기 위해 사용할 예측 모델.

2. **이상치 탐지**:
   - `IsolationForest(contamination=0.05, random_state=42)`:
     - `contamination=0.05`: 데이터의 5%를 이상치로 간주합니다.
     - `random_state=42`: 결과 재현성을 보장하기 위해 랜덤 시드를 고정합니다.
   - `clf.fit_predict(X_train_scaled)`:
     - 훈련 데이터를 사용해 이상치(-1)와 정상 데이터(1)를 감지합니다.
     - 결과는 `anomaly`라는 새로운 열로 추가됩니다.

3. **이상치와 정상 데이터 분리**:
   - `outliers`:
     - `X_train_scaled`에서 `anomaly == -1`인 행(이상치)만 선택.
   - `non_outliers`:
     - `X_train_scaled`에서 `anomaly != -1`인 행(정상 데이터)만 선택.

4. **이상치 대체**:
   - 각 특성(`col`)에 대해, 해당 열을 제외한 나머지 열(`features`)로 정상 데이터를 학습하여 이상치를 대체합니다.
   - `RandomForestRegressor`:
     - 비선형 관계를 잘 처리하는 모델로, 정상 데이터를 학습하고 이상치를 예측합니다.
   - `model.fit(clean_features, target)`:
     - 정상 데이터에서 현재 특성을 예측하기 위한 모델 학습.
   - `model.predict(outliers[features])`:
     - 이상치에 대해 예측 값을 생성하고, 이를 대체 값으로 사용.

5. **`anomaly` 열 제거**:
   - 이상치 처리가 완료되면 `anomaly` 열은 더 이상 필요 없으므로 삭제.


#### 사용 목적

1. **이상치 감지**:
   - 데이터 내 이상치를 탐지하고, 이상치가 모델에 미치는 부정적 영향을 줄이기 위해 처리합니다.

2. **이상치 대체**:
   - 이상치를 단순히 제거하지 않고, 대체 값(예측 값)을 생성해 데이터의 일관성을 유지합니다.

3. **데이터 정제**:
   - 학습 데이터를 정제해 모델의 학습 및 예측 성능을 향상시킵니다.


#### 활용 사례

1. **데이터 정제가 필요한 경우**:
   - 실험 또는 측정 데이터에서 비정상적인 값(센서 오류, 입력 오류 등)이 포함된 경우.

2. **모델 성능 향상**:
   - 이상치를 제거하거나 대체하지 않으면 모델이 데이터의 패턴을 잘 학습하지 못할 수 있습니다.

3. **특성 간 관계 활용**:
   - 다른 특성을 기반으로 이상치 값을 추정하고 대체하는 데 유용합니다.


#### 매개변수 설명

1. **`IsolationForest`**:
   - `contamination`: 데이터에서 이상치 비율(기본값: `'auto'`). 여기서는 5%로 설정.
   - `random_state`: 랜덤 시드로 결과 재현성을 보장.
   - `n_estimators`: 이상치 탐지를 위한 하위 샘플의 트리 개수(기본값: 100).

2. **`RandomForestRegressor`**:
   - `n_estimators`: 의사결정 나무의 개수(여기서는 100).
   - `random_state`: 랜덤 시드로 결과 재현성을 보장.


#### 사용 시 주의사항

1. **훈련 데이터에서만 이상치 탐지**:
   - 이상치 탐지는 훈련 데이터에서만 수행해야 합니다. 테스트 데이터는 훈련 데이터에 맞춘 변환만 적용합니다.

2. **이상치 대체가 필요한 경우**:
   - 이상치를 제거할 경우 데이터가 부족해질 수 있으므로, 대체가 필요한 상황을 잘 판단해야 합니다.

3. **이상치 비율(`contamination`) 설정**:
   - 이상치 비율이 실제 데이터 분포와 다르면 잘못된 이상치 탐지가 발생할 수 있습니다. 적절한 비율을 설정하는 것이 중요합니다.


#### 활용 상황

1. **이상치 감지만 필요한 경우**:
   ```python
   clf = IsolationForest(contamination=0.1, random_state=42)
   anomalies = clf.fit_predict(X_train_scaled)
   ```
   - 이상치 여부만 탐지하고, 값을 대체하지 않을 때.

2. **이상치 제거**:
   ```python
   X_train_cleaned = X_train_scaled[anomalies != -1]
   ```
   - 이상치를 직접 제거하고 데이터 크기를 줄이는 경우.

3. **이상치 대체**:
   - 코드에 작성된 방식처럼, 이상치 데이터를 다른 특성을 기반으로 예측 값을 생성하여 대체.

