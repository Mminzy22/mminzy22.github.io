---
layout: post
title: "Machine Learning 37: 과제: 주택 가격 예측 모델 구축(이상치 처리)"
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


## 1. 데이터 정규화 또는 표준화

### 정규화

   ```python
   # 정규화 추가
   from sklearn.preprocessing import MinMaxScaler

   # 정규화 (Normalization)
   scaler = MinMaxScaler()
   df_normalized = df.copy()
   columns_to_normalize = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
   df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

   # 정규화된 데이터 사용
   df = df_normalized
   ```

### 표준화

   ```python
   # 표준화 추가
   from sklearn.preprocessing import StandardScaler

   # 표준화 (Standardization)
   scaler = StandardScaler()
   df_standardized = df.copy()
   columns_to_standardize = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
   df_standardized[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

   # 표준화된 데이터 사용
   df = df_standardized
   ```

## 2. 이상치 처리

### IsolationForest와 LinearRegression을 조합한 이상치 대체체

   ```python
   from sklearn.ensemble import IsolationForest
   from sklearn.linear_model import LinearRegression

   # 1. 이상치 탐지 (Isolation Forest)
   clf = IsolationForest(contamination=0.05, random_state=42)
   df['anomaly'] = clf.fit_predict(df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']])

   # 2. 이상치와 정상 데이터 분리
   outliers = df[df['anomaly'] == -1]  # 이상치 행
   non_outliers = df[df['anomaly'] != -1]  # 정상 행

   # 3. 각 독립 변수에 대해 이상치 대체
   independent_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

   for col in independent_columns:
      # 현재 열(col)을 제외한 나머지 열(features)에서 이상치 제거
      features = [c for c in independent_columns if c != col]
      clean_features = non_outliers[features]  # 정상 데이터에서만 추출
      target = non_outliers[col]  # 현재 예측 대상 열의 정상 값
      
      # 모델 학습
      model = LinearRegression()
      model.fit(clean_features, target)
      
      # 이상치 대체 (예측)
      predicted_values = model.predict(outliers[features])
      df.loc[outliers.index, col] = predicted_values  # 이상치 열 대체

   ```

**1. 라이브러리 임포트**

```python
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
```

- **`IsolationForest`**:
  - 이상치를 탐지하기 위한 머신러닝 알고리즘.
  - 데이터를 분리하는 방식으로 이상치를 탐지하며, 비지도 학습 방식으로 작동.
  - `contamination` 파라미터로 이상치 비율을 설정.

- **`LinearRegression`**:
  - 독립 변수들로부터 특정 독립 변수(열)의 값을 예측하기 위한 선형 회귀 모델.
  - 정규 분포를 가정하며, 이상치를 예측하여 대체하는 데 사용.


**2. 이상치 탐지 (Isolation Forest)**

```python
clf = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = clf.fit_predict(df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']])
```

**동작**
- `IsolationForest` 모델이 `df`의 독립 변수들(`CRIM`, `ZN`, ..., `LSTAT`)을 입력으로 학습합니다.
- **`fit_predict()`**:
  - 입력 데이터를 학습하고, 이상치 여부를 반환.
  - 반환 값:
    - `1`: 정상 데이터.
    - `-1`: 이상치 데이터.

**결과**
- 새로운 열 `anomaly`가 추가됩니다:
  - `df['anomaly']`: 각 행이 이상치인지 정상인지 표시.

**설정된 파라미터**
- `contamination=0.05`: 데이터의 약 5%를 이상치로 간주.
- `random_state=42`: 결과 재현성을 보장.


**3. 이상치와 정상 데이터 분리**

```python
outliers = df[df['anomaly'] == -1]  # 이상치 행
non_outliers = df[df['anomaly'] != -1]  # 정상 행
```

**동작**
- **`outliers`**:
  - `anomaly` 값이 `-1`인 행(이상치)을 추출하여 새로운 데이터프레임으로 저장.
- **`non_outliers`**:
  - `anomaly` 값이 `1`인 행(정상 데이터)을 추출하여 새로운 데이터프레임으로 저장.


**4. 각 독립 변수에 대해 이상치 대체**

**코드 설명**

```python
independent_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
```

- `independent_columns`: 이상치를 탐지하고 대체해야 하는 독립 변수 목록.


**코드 동작 (루프)**

```python
for col in independent_columns:
    # 현재 열(col)을 제외한 나머지 열(features)에서 이상치 제거
    features = [c for c in independent_columns if c != col]
    clean_features = non_outliers[features]  # 정상 데이터에서만 추출
    target = non_outliers[col]  # 현재 예측 대상 열의 정상 값
```

1. **`features`**:
   - 현재 반복 중인 열(`col`)을 제외한 나머지 독립 변수 목록.
   - 예: `col = 'CRIM'`일 경우, `features = ['ZN', 'INDUS', ..., 'LSTAT']`.

2. **`clean_features`**:
   - 정상 데이터(`non_outliers`)에서만 입력 특성(`features`)을 추출.
   - 이상치를 포함하지 않은 데이터만 사용.

3. **`target`**:
   - 정상 데이터(`non_outliers`)에서 현재 열(`col`) 값을 예측 대상으로 설정.

**선형 회귀 모델 학습**

```python
model = LinearRegression()
model.fit(clean_features, target)
```

- **`clean_features`**: 입력 특성 데이터 (정상 데이터에서만 추출).
- **`target`**: 예측 대상 (정상 데이터에서 현재 열의 값).

**예시**
- `col = 'CRIM'`일 경우:
  - `clean_features`: `non_outliers[['ZN', 'INDUS', ..., 'LSTAT']]`.
  - `target`: `non_outliers['CRIM']`.


**이상치 대체**

```python
predicted_values = model.predict(outliers[features])
df.loc[outliers.index, col] = predicted_values
```

1. **`outliers[features]`**:
   - 이상치 데이터에서 현재 열(`col`)을 제외한 나머지 열(`features`)을 입력으로 사용.

2. **`predicted_values`**:
   - 학습된 선형 회귀 모델이 이상치 데이터를 입력받아 예측한 값.

3. **`df.loc[outliers.index, col] = predicted_values`**:
   - 예측된 값(`predicted_values`)을 `df`의 이상치 데이터에 대체.


**5. 결과 데이터프레임**
- 이상치가 정상 데이터 기반의 예측값으로 대체됩니다.
- `df`는 모든 독립 변수(`CRIM`, `ZN`, ..., `LSTAT`)에서 이상치가 제거된 상태로 갱신됩니다.


**이 코드의 강점**
1. **유연성**:
   - 독립 변수의 이상치를 각 변수마다 개별적으로 탐지하고 대체.
   - 여러 독립 변수의 이상치를 효과적으로 처리 가능.

2. **정확성**:
   - 정상 데이터를 기반으로 선형 회귀 모델을 학습하여 이상치를 대체.
   - 이상치 대체가 데이터의 전체적인 분포에 적합.

3. **효율성**:
   - `IsolationForest`와 `LinearRegression`을 조합하여 이상치를 탐지하고 대체하는 효율적인 방법.

**개선 가능성**
1. **모델 선택**:
   - 선형 회귀 대신 비선형 모델(예: `RandomForestRegressor`)을 사용하면 더 정교한 대체가 가능.

2. **스케일링**:
   - 데이터의 분포가 크게 다를 경우, 이상치 탐지 및 예측 전에 정규화(스케일링)를 추가하면 성능 향상 가능.

3. **이상치 탐지 범위**:
   - 특정 독립 변수마다 `IsolationForest`의 `contamination` 값을 개별적으로 설정하면 더 정밀한 탐지가 가능.

### RandomForestRegressor를 사용한 이상치 대체

`RandomForestRegressor`는 비선형 회귀 모델로, 선형 회귀보다 복잡한 데이터 분포를 잘 학습할 수 있습니다.

   ```python
   from sklearn.ensemble import IsolationForest, RandomForestRegressor

   # 1. 이상치 탐지 (Isolation Forest)
   clf = IsolationForest(contamination=0.05, random_state=42)
   df['anomaly'] = clf.fit_predict(df_imputed[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']])

   # 2. 이상치와 정상 데이터 분리
   outliers = df_imputed[df_imputed['anomaly'] == -1]  # 이상치 행
   non_outliers = df_imputed[df_imputed['anomaly'] != -1]  # 정상 행

   # 3. 각 독립 변수에 대해 이상치 대체
   independent_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

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
      df_imputed.loc[outliers.index, col] = predicted_values  # 이상치 열 대체
   ```


 **코드 설명**

**1. Isolation Forest로 이상치 탐지**

```python
clf = IsolationForest(contamination=0.05, random_state=42)
df_imputed['anomaly'] = clf.fit_predict(df_imputed[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']])
```

- `IsolationForest`를 사용해 이상치를 탐지하고, `anomaly` 열에 결과를 저장합니다.
  - `-1`: 이상치.
  - `1`: 정상 데이터.


**2. 이상치와 정상 데이터 분리**

```python
outliers = df_imputed[df_imputed['anomaly'] == -1]
non_outliers = df_imputed[df_imputed['anomaly'] != -1]
```

- **`outliers`**: 이상치 데이터만 포함.
- **`non_outliers`**: 정상 데이터만 포함.


**3. RandomForestRegressor로 이상치 대체**

**독립 변수와 대체 대상 설정**

```python
features = [c for c in independent_columns if c != col]
clean_features = non_outliers[features]
target = non_outliers[col]
```

- **`features`**: 현재 반복 중인 열을 제외한 나머지 독립 변수.
- **`clean_features`**: 정상 데이터에서 입력 변수(`features`).
- **`target`**: 정상 데이터에서 현재 열의 값.

**RandomForestRegressor 모델 학습**

```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(clean_features, target)
```

- **`RandomForestRegressor`**:
  - `n_estimators=100`: 100개의 결정 트리를 사용하는 랜덤 포레스트 회귀.
  - `random_state=42`: 재현 가능한 결과를 위해 난수 고정.

**이상치 예측**

```python
predicted_values = model.predict(outliers[features])
```

- 이상치 데이터(`outliers`)의 독립 변수(`features`)를 입력으로 사용해 값을 예측.

**이상치 대체**

```python
df_imputed.loc[outliers.index, col] = predicted_values
```

- 예측된 값을 이상치 위치에 대체.


 **추가 설정**

1. 하이퍼파라미터 튜닝
`RandomForestRegressor`의 성능을 개선하기 위해 하이퍼파라미터를 조정할 수 있습니다:

```python
model = RandomForestRegressor(
    n_estimators=200,      # 더 많은 트리 사용
    max_depth=10,          # 트리의 최대 깊이 제한
    min_samples_split=5,   # 분할을 위한 최소 샘플 수
    random_state=42
)
```

2. 데이터 스케일링
`RandomForestRegressor`는 스케일에 민감하지 않지만, 다른 모델과 조합해 사용하려면 데이터 정규화를 고려할 수 있습니다:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_imputed[independent_columns])
df_imputed[independent_columns] = scaled_features
```

**결과 확인**

(1) 대체된 이상치 확인

```python
print("이상치 대체 후 데이터:")
print(df_imputed.head())
```

(2) 데이터 시각화

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 이상치 대체 후 분포 확인
sns.boxplot(data=df_imputed[independent_columns])
plt.title('Feature Distribution After Outlier Replacement')
plt.show()
```
