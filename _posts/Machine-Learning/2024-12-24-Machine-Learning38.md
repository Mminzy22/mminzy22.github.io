---
layout: post
title: "Machine Learning 38: 과제: 주택 가격 예측 모델 구축(모델 학습 및 평가)"
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


### 1. 데이터 준비 및 분리

이 단계는 데이터 분석의 기본적인 시작 단계로, 모델 학습과 평가를 위해 데이터를 준비하고 적절히 분리하는 과정입니다.

#### 1. **데이터 준비**

```python
X = df.drop(columns=['MEDV']).values
y = df['MEDV'].values
```

- **독립 변수(특징)**:
  - `X`는 예측에 필요한 독립 변수(특징)를 의미합니다.
  - `df.drop(columns=['MEDV'])`를 통해 목표 변수(`MEDV`)를 제외한 모든 열을 선택합니다.

- **종속 변수(목표값)**:
  - `y`는 모델이 예측할 목표 변수로, 주택 가격 데이터를 나타냅니다.
  - `df['MEDV']`를 통해 `MEDV` 열을 추출합니다.

- **`.values`**:
  - 데이터를 Pandas 데이터프레임에서 Numpy 배열 형식으로 변환합니다.
  - 머신러닝 모델은 Numpy 배열 형태의 입력을 요구하므로 변환이 필요합니다.


#### 2. **데이터 분리**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **`train_test_split`**:
  - 데이터를 학습용 데이터(`train`)와 테스트용 데이터(`test`)로 나눕니다.
  - 이를 통해 모델이 학습한 데이터 외에 테스트 데이터를 사용하여 평가할 수 있습니다.

##### 주요 매개변수:
1. **`X`, `y`**:
   - `X`: 독립 변수(특징 행렬).
   - `y`: 종속 변수(목표값 벡터).

2. **`test_size=0.2`**:
   - 전체 데이터의 20%를 테스트 데이터로 사용합니다.
   - 나머지 80%는 학습 데이터로 사용됩니다.

3. **`random_state=42`**:
   - 데이터를 섞는 순서를 고정하여 실험 재현성을 보장합니다.
   - 동일한 `random_state` 값을 사용하면, 실행할 때마다 동일한 데이터 분할 결과를 얻을 수 있습니다.

##### 반환값:
- **`X_train`**:
  - 학습 데이터의 독립 변수(특징).
- **`X_test`**:
  - 테스트 데이터의 독립 변수(특징).
- **`y_train`**:
  - 학습 데이터의 종속 변수(목표값).
- **`y_test`**:
  - 테스트 데이터의 종속 변수(목표값).


#### 3. **데이터 분리 목적**

데이터를 학습과 테스트로 분리하는 이유는 다음과 같습니다:

1. **모델의 일반화 성능 평가**:
   - 모델이 학습 데이터에 과도하게 적합(overfitting)하지 않고, 새로운 데이터에서도 잘 작동하는지 확인하기 위함입니다.
   - 테스트 데이터는 모델이 학습하지 않은 데이터로, 모델의 예측 능력을 객관적으로 평가할 수 있습니다.

2. **학습과 검증의 독립성 유지**:
   - 학습 데이터(`X_train`, `y_train`)는 모델의 학습에만 사용되며, 테스트 데이터(`X_test`, `y_test`)는 평가에만 사용됩니다.
   - 이로 인해 데이터가 섞여 모델 성능이 과대평가되는 것을 방지할 수 있습니다.

3. **실제 환경 시뮬레이션**:
   - 테스트 데이터는 실제 예측 환경에서 모델이 새로운 데이터를 처리하는 능력을 시뮬레이션합니다.
   - 이를 통해 모델의 예측 성능을 미리 확인할 수 있습니다.


#### 4. **데이터 분리 비율 설정**
- **학습 데이터와 테스트 데이터의 비율**:
  - 일반적으로 테스트 데이터는 전체 데이터의 20~30% 정도로 설정합니다.
  - 데이터가 많을수록 테스트 데이터 비율을 줄일 수 있습니다.
  - 데이터가 적을 경우, 테스트 데이터를 너무 적게 설정하면 평가가 불안정해질 수 있습니다.

- **검증 데이터 필요 시**:
  - 데이터가 충분히 많다면, 학습 데이터에서 검증 데이터를 추가로 분리하여 모델 성능을 미리 평가할 수 있습니다.
  - 이 경우 일반적으로 학습:검증:테스트 데이터를 60:20:20 비율로 나누기도 합니다.


머신러닝 모델 학습과 평가의 기반을 마련하는 중요한 과정으로, 데이터 분리 방식에 따라 모델의 성능 평가가 크게 달라질 수 있습니다.

### 2. 선형 회귀 모델 평가

이 단계에서는 **선형 회귀 모델(Linear Regression)**을 학습시키고, 이를 테스트 데이터로 평가하여 성능 지표를 계산하는 과정을 다룹니다. 선형 회귀는 데이터를 가장 잘 설명할 수 있는 직선(또는 초평면)을 찾아내는 회귀 분석 기법입니다.


#### 1. **모델 생성 및 학습**

```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```

- **`LinearRegression()`**:
  - Scikit-learn에서 제공하는 선형 회귀 모델 클래스입니다.
  - 독립 변수와 종속 변수 사이의 선형 관계를 학습합니다.
  - 이 모델은 다음과 같은 수식을 사용합니다:
    \\[
    y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
    \\]
    여기서:
    - \\( y \\): 종속 변수(예측값).
    - \\( x_1, x_2, \ldots, x_n \\): 독립 변수(특징).
    - \\( \beta_0 \\): 절편(intercept).
    - \\( \beta_1, \beta_2, \ldots, \beta_n \\): 가중치(기울기).

- **`fit(X_train, y_train)`**:
  - 학습 데이터를 사용해 모델을 학습합니다.
  - 학습 과정에서 선형 회귀 모델은 오차를 최소화하는 \\( \beta \\) 값을 계산합니다.
  - 오차는 일반적으로 **최소제곱법(Least Squares Method)**으로 계산됩니다:
    \\[
    \text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
    \\]
    - \\( y_i \\): 실제값.
    - \\( \hat{y}_i \\): 예측값.
    - 모델은 이 RSS를 최소화하는 방향으로 \\( \beta \\)를 학습합니다.


#### 2. **테스트 데이터 예측**

```python
y_pred_lr = lr_model.predict(X_test)
```

- **`predict(X_test)`**:
  - 학습된 모델을 사용하여 테스트 데이터(`X_test`)의 종속 변수(목표값)를 예측합니다.
  - 반환값 `y_pred_lr`는 예측값의 배열입니다.


#### 3. **성능 평가**

```python
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)
```

- **평가 지표**:
  선형 회귀 모델의 성능은 여러 평가 지표를 통해 확인할 수 있습니다.

1. **MAE (Mean Absolute Error)**:

   \\[
   \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
   \\]
   
   - 실제값과 예측값의 차이의 절대값 평균.
   - 값이 작을수록 모델이 예측에 성공한 것입니다.
   - 단위는 목표값과 동일합니다.

2. **MSE (Mean Squared Error)**:

   \\[
   \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   \\]

   - 예측값과 실제값의 차이를 제곱한 뒤 평균을 구한 값.
   - 큰 오차에 더 큰 패널티를 부여합니다.
   - 값이 작을수록 좋습니다.

3. **R² (결정 계수)**:

   $$
   R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
   $$

   - \\(\text{SS}_{\text{res}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2\\): 잔차 제곱합(Residual Sum of Squares).
   - \\(\text{SS}_{\text{tot}} = \sum_{i=1}^{n} (y_i - \bar{y})^2\\): 총 제곱합(Total Sum of Squares).
   - 값의 범위는 \\(-\infty\\)에서 1까지이며, 1에 가까울수록 모델이 데이터를 잘 설명합니다.
   - 음수 값은 모델이 데이터를 잘 설명하지 못함을 의미합니다.


#### 4. **교차 검증을 통한 성능 평가**

```python
cv_results_lr = cross_validate(lr_model, X_train, y_train, cv=3, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'])
```

- **교차 검증 (Cross-Validation)**:
  - 데이터를 여러 개의 **폴드(fold)**로 나눠 모델을 반복 학습 및 평가하여 성능을 안정적으로 측정합니다.
  - **`cv=3`**:
    - 데이터를 3등분하여, 한 번은 테스트용, 나머지는 학습용으로 사용합니다.
    - 이 과정을 3번 반복하여 평균 성능을 계산합니다.

- **`scoring`**:
  - `neg_mean_squared_error`: MSE를 음수 형태로 반환.
  - `neg_mean_absolute_error`: MAE를 음수 형태로 반환.
  - `r2`: 결정 계수를 반환.


#### 5. **결과 출력**

```python
print("Linear Regression Cross-Validation Results:")
print(f"Mean CV MSE: {-cv_results_lr['test_neg_mean_squared_error'].mean():.4f}")
print(f"Mean CV MAE: {-cv_results_lr['test_neg_mean_absolute_error'].mean():.4f}")
print(f"Mean CV R^2: {cv_results_lr['test_r2'].mean():.4f}")
```

- **출력 해석**:
  - **Mean CV MSE**:
    - 교차 검증을 통해 얻은 MSE의 평균값.
    - 값이 작을수록 좋습니다.
  - **Mean CV MAE**:
    - 교차 검증으로 계산된 MAE의 평균값.
    - 예측 오차의 크기를 확인할 수 있습니다.
  - **Mean CV R²**:
    - 교차 검증을 통해 얻은 R²의 평균값.
    - 1에 가까울수록 모델이 데이터를 잘 설명합니다.


#### 이 단계의 핵심
1. **선형 회귀의 기본 원리**:
   - 데이터를 설명할 수 있는 가장 단순한 선형 모델을 구축합니다.

2. **평가 지표 활용**:
   - MAE, MSE, R² 등 다양한 지표를 통해 모델의 예측 성능을 다각도로 평가합니다.

3. **교차 검증**:
   - 데이터 분할 및 평가를 반복 수행하여 모델 성능을 안정적으로 측정합니다.

4. **결과 해석**:
   - 선형 회귀는 단순하고 빠르게 계산되지만, 복잡한 데이터에서 성능이 제한적일 수 있습니다.
   - 이후 다른 모델과의 비교를 통해 성능 개선 가능성을 탐색합니다.

### 3. 랜덤 포레스트 하이퍼파라미터 튜닝

이 단계에서는 랜덤 포레스트(Random Forest) 모델의 하이퍼파라미터를 튜닝하여 성능을 최적화하는 과정을 다룹니다. 하이퍼파라미터는 모델 학습 전에 설정해야 하는 값으로, 모델의 예측 성능에 큰 영향을 미칩니다.


#### 1. **랜덤 포레스트 개요**
랜덤 포레스트는 앙상블 학습 기법으로, 다수의 결정 트리를 결합하여 예측합니다. 각 트리는 훈련 데이터의 일부(부트스트랩 샘플링)와 일부 특징을 사용해 생성됩니다. 최종 예측은 모든 트리의 평균 또는 다수결로 결정됩니다.

랜덤 포레스트는 다음과 같은 주요 하이퍼파라미터를 제공합니다:
- **`n_estimators`**: 생성할 트리의 개수.
- **`max_depth`**: 각 트리의 최대 깊이.
- **`min_samples_split`**: 노드를 분할하는 데 필요한 최소 샘플 수.
- **`min_samples_leaf`**: 리프 노드에 있어야 하는 최소 샘플 수.


#### 2. **하이퍼파라미터 설정**

```python
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

##### 주요 파라미터 설명:
1. **`n_estimators`**:
   - 랜덤 포레스트에서 생성할 트리의 개수입니다.
   - 일반적으로 클수록 성능이 안정적이지만, 학습 시간은 증가합니다.
   - 테스트 설정: `[50, 100, 200]`.

2. **`max_depth`**:
   - 각 트리의 최대 깊이를 설정합니다.
   - 너무 깊으면 과적합이 발생할 수 있으므로, 적절한 값을 찾는 것이 중요합니다.
   - 테스트 설정: `[5, 10, None]` (`None`은 깊이 제한이 없음을 의미).

3. **`min_samples_split`**:
   - 노드를 분할하는 데 필요한 최소 샘플 수입니다.
   - 값이 클수록 모델이 더 단순해지고, 과적합 가능성이 줄어듭니다.
   - 테스트 설정: `[2, 5, 10]`.

4. **`min_samples_leaf`**:
   - 리프 노드에 있어야 하는 최소 샘플 수입니다.
   - 값이 클수록 모델의 복잡도가 줄어듭니다.
   - 테스트 설정: `[1, 2, 4]`.


#### 3. **GridSearchCV로 하이퍼파라미터 튜닝**

```python
rf_grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=-1
)
```

##### 주요 매개변수 설명:
1. **`RandomForestRegressor`**:
   - 랜덤 포레스트 회귀 모델입니다.
   - `random_state=42`를 설정하여 실험의 재현성을 보장합니다.

2. **`param_grid`**:
   - 하이퍼파라미터 검색 범위를 지정합니다.
   - 모든 가능한 조합을 시도하여 최적의 조합을 찾습니다.

3. **`scoring='neg_mean_squared_error'`**:
   - 평균 제곱 오차(MSE)를 기반으로 모델 성능을 평가합니다.
   - 음수 형태로 제공되므로 결과를 해석할 때는 `-`를 곱해 양수로 변환합니다.

4. **`cv=3`**:
   - 3-폴드 교차 검증을 수행합니다.
   - 데이터를 3등분하여 학습과 검증을 반복하며 평균 성능을 계산합니다.

5. **`verbose=2`**:
   - 검색 과정에서 상세한 진행 상황을 출력합니다.

6. **`n_jobs=-1`**:
   - 모든 CPU 코어를 사용해 병렬 처리를 수행합니다.
   - 검색 속도를 크게 향상시킵니다.


#### 4. **모델 학습**

```python
rf_grid_search.fit(X_train, y_train)
```

- **`fit`**:
  - GridSearchCV가 `param_grid`에 지정된 모든 파라미터 조합을 시도하며 최적의 조합을 찾습니다.
  - 교차 검증을 통해 각 조합의 평균 성능을 계산합니다.


#### 5. **최적 하이퍼파라미터 및 성능 확인**

```python
print("Best parameters for RandomForest: ", rf_grid_search.best_params_)
print("Best score for RandomForest: ", -rf_grid_search.best_score_)
```

- **`best_params_`**:
  - 검색된 하이퍼파라미터 중 모델 성능이 가장 높은 조합을 반환합니다.

- **`best_score_`**:
  - 최적 하이퍼파라미터 조합의 교차 검증 성능을 반환합니다.
  - MSE의 음수 값으로 제공되므로, 해석할 때 `-`를 곱하여 양수로 변환합니다.


#### 6. **결과 출력 예시**

```plaintext
Best parameters for RandomForest: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
Best score for RandomForest: 23.4578
```

- **해석**:
  - 최적 하이퍼파라미터 조합:
    - `n_estimators=200`: 200개의 트리를 사용.
    - `max_depth=10`: 각 트리의 최대 깊이는 10.
    - `min_samples_split=5`: 노드 분할을 위해 최소 5개의 샘플이 필요.
    - `min_samples_leaf=2`: 리프 노드에 최소 2개의 샘플이 있어야 함.
  - 최적 교차 검증 점수(MSE): 약 23.4578.


#### 이 단계의 핵심

1. **랜덤 포레스트의 장점**:
   - 비선형 관계를 잘 처리하고, 과적합 가능성이 낮으며, 다양한 데이터에 강건함.

2. **하이퍼파라미터 튜닝 필요성**:
   - 랜덤 포레스트의 성능은 하이퍼파라미터 설정에 따라 크게 달라집니다.
   - GridSearchCV를 사용하여 최적의 하이퍼파라미터 조합을 자동으로 탐색합니다.

3. **병렬 처리 활용**:
   - `n_jobs=-1`를 사용하여 검색 속도를 높입니다.

4. **결과 해석**:
   - 최적의 하이퍼파라미터와 교차 검증 성능을 확인하고, 이후 이 최적 모델로 테스트 데이터를 평가합니다.


### 4. 그래디언트 부스팅 하이퍼파라미터 튜닝

그래디언트 부스팅(Gradient Boosting)은 **부스팅(Boosting)** 기법의 하나로, 약한 학습자(weak learner) 여러 개를 순차적으로 학습하여 강한 학습자(strong learner)를 만드는 앙상블 모델입니다. 랜덤 포레스트와 달리 트리들이 순차적으로 학습되며, 이전 단계의 오차를 보정합니다.


#### 1. **그래디언트 부스팅 모델 개요**
- 각 트리는 이전 트리의 예측 오차(residuals)를 기반으로 학습합니다.
- 오차를 줄이기 위해 **그레디언트 하강법**(Gradient Descent)을 사용합니다.
- 주로 **회귀**와 **분류** 문제에서 높은 성능을 보입니다.


#### 2. **하이퍼파라미터 설정**

```python
gb_param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.7, 0.3)
}
```

##### 주요 하이퍼파라미터 설명:
1. **`n_estimators`**:
   - 생성할 트리의 개수.
   - 트리 개수가 많으면 모델 복잡도가 증가하여 과적합 가능성이 있지만, 적으면 성능이 낮아질 수 있습니다.
   - 테스트 설정: 랜덤 정수 `[50, 200]`.

2. **`max_depth`**:
   - 각 트리의 최대 깊이.
   - 깊이가 깊을수록 모델의 표현력은 증가하지만, 과적합 가능성도 높아집니다.
   - 테스트 설정: 랜덤 정수 `[3, 10]`.

3. **`learning_rate`**:
   - 각 트리가 학습하는 정도를 조정하는 값.
   - 값이 작으면 학습 속도가 느려지지만, 일반화 성능이 좋아질 수 있습니다.
   - 테스트 설정: 균일 분포에서 무작위 샘플링 `[0.01, 0.3]`.

4. **`subsample`**:
   - 각 트리가 학습할 때 사용할 샘플의 비율.
   - 1.0이면 모든 샘플을 사용하며, 0.7~1.0이면 일부 샘플만 사용합니다.
   - 테스트 설정: 균일 분포에서 무작위 샘플링 `[0.7, 1.0]`.


#### 3. **RandomizedSearchCV로 하이퍼파라미터 탐색**

```python
gb_random_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=gb_param_dist,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
```

##### 주요 매개변수 설명:
1. **`GradientBoostingRegressor`**:
   - 그래디언트 부스팅 회귀 모델.
   - 이전 트리의 오차를 학습하여 성능을 향상시킵니다.

2. **`param_distributions`**:
   - 탐색할 하이퍼파라미터 범위를 정의합니다.
   - 무작위 샘플링 방식으로 조합을 시도합니다.

3. **`n_iter=50`**:
   - 50개의 랜덤 하이퍼파라미터 조합을 시도합니다.
   - 탐색 공간이 클 경우 효율적인 접근법입니다.

4. **`scoring='neg_mean_squared_error'`**:
   - 평균 제곱 오차(MSE)를 사용하여 모델 성능을 평가합니다.
   - 음수 값으로 반환되므로, 결과 해석 시 `-`를 곱해 양수로 변환합니다.

5. **`cv=3`**:
   - 3-폴드 교차 검증을 통해 각 조합의 평균 성능을 계산합니다.

6. **`verbose=2`**:
   - 탐색 진행 상황을 자세히 출력합니다.

7. **`n_jobs=-1`**:
   - 병렬 처리를 통해 탐색 속도를 높입니다.
   - 모든 CPU 코어를 사용합니다.

8. **`random_state=42`**:
   - 랜덤성을 고정하여 실행할 때마다 동일한 결과를 얻을 수 있습니다.


#### 4. **모델 학습**

```python
gb_random_search.fit(X_train, y_train)
```

- **`fit`**:
  - `RandomizedSearchCV`가 하이퍼파라미터 조합을 무작위로 선택하여 모델을 학습합니다.
  - 교차 검증을 통해 각 조합의 평균 성능을 계산하고, 최적의 하이퍼파라미터를 찾습니다.


#### 5. **최적 하이퍼파라미터 및 성능 확인**

```python
print("Best parameters for GradientBoosting: ", gb_random_search.best_params_)
print("Best score for GradientBoosting: ", -gb_random_search.best_score_)
```

- **`best_params_`**:
  - 검색된 하이퍼파라미터 중 성능이 가장 좋은 조합을 반환합니다.

- **`best_score_`**:
  - 최적 하이퍼파라미터 조합의 교차 검증 평균 성능(MSE)을 반환합니다.
  - 음수 값으로 반환되므로, 결과 해석 시 `-`를 곱해 양수로 변환합니다.


#### 6. **결과 출력 예시**

```plaintext
Best parameters for GradientBoosting: {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.07, 'subsample': 0.8}
Best score for GradientBoosting: 18.2345
```

- **해석**:
  - 최적 하이퍼파라미터 조합:
    - `n_estimators=150`: 150개의 트리를 생성.
    - `max_depth=5`: 각 트리의 최대 깊이는 5.
    - `learning_rate=0.07`: 트리의 학습률은 0.07.
    - `subsample=0.8`: 트리 학습에 전체 데이터의 80%만 사용.
  - 최적 교차 검증 점수(MSE): 약 18.2345.


#### 이 단계의 핵심

1. **그래디언트 부스팅의 특징**:
   - 이전 모델의 오차를 보완하여 성능을 점진적으로 개선합니다.
   - 하이퍼파라미터 튜닝을 통해 모델의 성능과 과적합 방지 간의 균형을 맞춥니다.

2. **RandomizedSearchCV 사용 이유**:
   - 무작위 탐색은 하이퍼파라미터 공간이 크고 복잡할 때 효율적입니다.
   - GridSearchCV보다 계산 비용이 낮습니다.

3. **평가 및 튜닝**:
   - 최적의 하이퍼파라미터 조합을 찾고, 모델 성능을 교차 검증을 통해 안정적으로 평가합니다.

4. **결과 활용**:
   - 최적화된 그래디언트 부스팅 모델을 사용하여 테스트 데이터 또는 실제 데이터에서 예측 성능을 평가하고 활용할 수 있습니다.

### 5. 최적 모델 성능 평가

이 단계에서는 앞서 학습한 최적화된 모델(선형 회귀, 랜덤 포레스트, 그래디언트 부스팅)을 사용하여 테스트 데이터에 대한 성능을 비교합니다. 각 모델의 성능은 **MAE**, **MSE**, **R²** 지표를 통해 평가되며, 결과를 표로 정리하고 시각화합니다.


#### 1. **최적화된 모델 준비**

```python
final_models = {
    "LinearRegression": lr_model,
    "RandomForest": rf_grid_search.best_estimator_,
    "GradientBoosting": gb_random_search.best_estimator_
}
```

- **`lr_model`**:
  - 학습된 선형 회귀 모델입니다.
  - 하이퍼파라미터 튜닝이 필요하지 않으므로, 기본 모델을 사용합니다.

- **`rf_grid_search.best_estimator_`**:
  - `GridSearchCV`를 통해 최적의 하이퍼파라미터를 가진 랜덤 포레스트 모델입니다.

- **`gb_random_search.best_estimator_`**:
  - `RandomizedSearchCV`를 통해 최적화된 그래디언트 부스팅 모델입니다.


#### 2. **테스트 데이터 예측 및 평가**

```python
results = {}
for name, model in final_models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "MSE": mse, "R^2": r2}
```

- **반복적으로 각 모델 평가**:
  - **`predict(X_test)`**: 테스트 데이터에 대해 예측값 생성.
  - **`mean_absolute_error(y_test, y_pred)`**: MAE 계산.
  - **`mean_squared_error(y_test, y_pred)`**: MSE 계산.
  - **`r2_score(y_test, y_pred)`**: R² 계산.

- **결과 저장**:
  - `results` 딕셔너리에는 각 모델의 성능 지표가 저장됩니다.
  - 딕셔너리 구조:
    ```python
    {
        "LinearRegression": {"MAE": value, "MSE": value, "R^2": value},
        "RandomForest": {"MAE": value, "MSE": value, "R^2": value},
        "GradientBoosting": {"MAE": value, "MSE": value, "R^2": value}
    }
    ```


#### 3. **결과 정리 및 출력**

```python
results_df = pd.DataFrame(results)
print(results_df)
```

- **DataFrame 생성**:
  - `results` 딕셔너리를 Pandas DataFrame으로 변환하여 가독성을 높입니다.
  - 행은 성능 지표(`MAE`, `MSE`, `R²`), 열은 모델 이름으로 구성됩니다.

- **출력 예시**:
  ```plaintext
                      LinearRegression  RandomForest  GradientBoosting
    MAE                   3.5123           2.8654           2.9521
    MSE                  18.4578          12.5673          13.4321
    R^2                   0.8456           0.9123           0.9045
  ```


#### 4. **결과 시각화**

```python
results_df.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.legend(title="Models")
plt.xticks(rotation=0)
plt.show()
```

- **막대 그래프 생성**:
  - 성능 지표별로 모델 성능을 비교하기 위해 막대 그래프를 생성합니다.
  - `kind='bar'`: 막대 그래프 생성.
  - `figsize=(10, 6)`: 그래프 크기 설정.

- **그래프 설정**:
  - **`title`**: 그래프 제목.
  - **`xlabel`**: x축 레이블(평가 지표).
  - **`ylabel`**: y축 레이블(지표 값).
  - **`legend`**: 그래프 범례(모델 이름).
  - **`xticks(rotation=0)`**: x축 레이블을 수평으로 표시.


#### 5. **결과 해석**

1. **MAE**:
   - 모델의 평균 절대 오차를 나타냅니다.
   - 값이 작을수록 모델의 예측값이 실제값에 가깝습니다.

2. **MSE**:
   - 평균 제곱 오차로, 큰 오차에 더 큰 페널티를 부여합니다.
   - 값이 작을수록 모델의 예측 성능이 우수합니다.

3. **R²**:
   - 결정 계수로, 모델이 데이터를 얼마나 잘 설명하는지 나타냅니다.
   - 1에 가까울수록 모델이 데이터를 잘 설명합니다.

##### 예시 해석:
- **랜덤 포레스트(RandomForest)**:
  - MSE와 R²에서 가장 높은 성능을 보여 가장 우수한 모델로 평가됩니다.
  - MAE 역시 가장 낮아, 실제값과의 오차가 가장 적습니다.

- **그래디언트 부스팅(GradientBoosting)**:
  - 랜덤 포레스트와 비슷한 성능을 보여, 데이터의 복잡성에 따라 더 나은 선택이 될 수 있습니다.

- **선형 회귀(LinearRegression)**:
  - 성능이 다른 두 모델보다 낮지만, 데이터가 선형적 관계를 가질 경우 간단하고 빠르게 결과를 도출할 수 있습니다.


#### 이 단계의 핵심

1. **최적 모델 선택**:
   - 테스트 데이터에서 가장 높은 성능을 보인 모델을 선택합니다.
   - 실무에서는 단순히 성능만으로 선택하지 않고, 모델의 해석 가능성, 계산 비용 등도 고려합니다.

2. **성능 비교**:
   - 모든 모델의 MAE, MSE, R² 값을 비교하여, 데이터에 가장 적합한 모델을 결정합니다.

3. **결과 시각화**:
   - 모델별 성능 차이를 시각적으로 확인하여, 데이터에 맞는 최적의 모델을 쉽게 선택할 수 있습니다.

4. **활용**:
   - 최종 선택된 모델을 사용하여 새로운 데이터에 대한 예측을 수행하거나, 실무 프로젝트에 적용합니다.
