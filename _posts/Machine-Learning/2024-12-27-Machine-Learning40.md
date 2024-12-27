---
layout: post
title: "Machine Learning 40: 과제: 고객 세분화 분석(클러스터링 기법 비교 및 시각화화)"
date: 2024-12-27
categories: [Machine Learning]
tag: []
---

### 과제 1번: 비지도학습

#### 주제: 고객 세분화 분석
- [고객 데이터셋]({{ site.baseurl }}/assets/downloads/Mall_Customers.csv)을 사용하여 비슷한 행동을 보이는 고객 그룹을 식별합니다.

<details>
<summary><strong>컬럼별 설명</strong></summary>
   <ol>
      <li>CustomerID: 각 고객을 고유하게 식별할 수 있는 ID입니다.</li>
      <li>Gender: 고객의 성별입니다. (예: "Male" 또는 "Female")</li>
      <li>Age: 고객의 나이를 나타냅니다.</li>
      <li>Annual Income (k$): 고객의 연간 수입을 1,000달러 단위로 나타냅니다.</li>
      <li>Spending Score (1-100): 고객의 쇼핑 점수로, 1부터 100 사이의 값입니다. 고객의 소비 습관이나 충성도를 평가하는 데 사용됩니다.</li>
   </ol>
</details>

<details>
<summary><strong>과제 가이드</strong></summary>
   <ul>
      <li>데이터셋 탐색 및 전처리:
         <ul>
            <li>결측치 처리</li>
            <li>스케일링
               <ul>
                  <li>데이터의 스케일을 조정하기 위해 표준화(Standardization) 또는 정규화(Normalization)를 수행합니다</li>
               </ul>
            </li>
         </ul>
      </li>
      <li>클러스터링 기법 적용:
         <ul>
            <li>K-means</li>
            <li>계층적 군집화</li>
            <li>DBSCAN 등의 알고리즘</li>
         </ul>
      </li>
      <li>최적의 클러스터 수 결정:
         <ul>
            <li>엘보우 방법 또는 실루엣 점수를 사용하여 최적의 클러스터 수를 찾습니다.</li>
         </ul>
      </li>
      <li>결과 시각화: 
         <ul>
            <li>클러스터링 결과를 2D 또는 3D로 시각화하여 고객 세분화 결과를 분석합니다.
               <ul>
                  <li>시각화: matplotlib 또는 seaborn을 사용하여 클러스터를 색상으로 구분하여 시각화합니다. 2D 플롯을 사용하여 각 클러스터를 다른 색으로 표현합니다.</li>
               </ul>
            </li>
         </ul>
      </li>
   </ul>
</details>

<details>
<summary><strong>도전 과제 가이드</strong></summary>
   <ul>
      <li>다양한 클러스터링 기법 비교 ⭐⭐⭐⭐
         <ul>
            <li>DBSCAN 외에 Gaussian Mixture Model(GMM)와 같은 다른 클러스터링 기법을 적용하고 성능을 비교합니다.</li>
         </ul>
      </li>
      <li>고객 행동 예측 모델 구축 ⭐⭐⭐⭐⭐
         <ul>
            <li>클러스터링 결과를 바탕으로 특정 클러스터에 속하는 고객의 행동을 예측하는 분류 모델을 구축합니다. 예를 들어, 구매 가능성을 예측하는 모델을 만들 수 있습니다.</li>
         </ul>
      </li>
      <li>시계열 분석⭐⭐⭐⭐
         <ul>
            <li>고객의 행동 변화를 시간에 따라 분석하여 특정 그룹의 트렌드를 시계열 데이터로 시각화합니다.</li>
         </ul>
      </li>
   </ul>
</details>

### 1. 다양한 클러스터링 기법 비교 및 최적의 클러스터 수 결정

```python
# 실루엣 점수를 저장할 딕셔너리
silhouette_scores = {
  "KMeans": {},
  "Hierarchical Clustering": {},
  "GMM": {},
  "DBSCAN": {}  # (eps, min_samples) 조합을 키로 사용
}

# 엘보우 방법을 위한 WCSS 저장 리스트
wcss = []

# 실루엣 점수 계산 및 저장
for k in range(2, 11):
  # KMeans
  km = KMeans(n_clusters=k, random_state=42)
  silhouette_scores["KMeans"][k] = silhouette_score(data, km.fit_predict(data))

  wcss.append(km.inertia_)

  # Hierarchical Clustering
  hc = AgglomerativeClustering(n_clusters=k)
  silhouette_scores["Hierarchical Clustering"][k] = silhouette_score(data, hc.fit_predict(data))

  # GMM
  gmm = GaussianMixture(n_components=k, random_state=42)
  silhouette_scores["GMM"][k] = silhouette_score(data, gmm.fit_predict(data))

  print(f"k={k}, KMeans={silhouette_scores['KMeans'][k]}, Hierarchical={silhouette_scores['Hierarchical Clustering'][k]}, GMM={silhouette_scores['GMM'][k]}")

# DBSCAN 파라미터 조합 설정
eps_values = [0.5, 0.6, 0.7]
min_samples_values = [6, 7, 8]

# DBSCAN 점수 계산
dbscan_scores = {}
for eps in eps_values:
  for min_samples in min_samples_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    data['Cluster_DBSCAN'] = dbscan.fit_predict(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']])
    
    # 노이즈 제거 (-1 레이블 제외)
    labels = data['Cluster_DBSCAN']
    if len(set(labels)) > 1:  # 최소 2개의 클러스터가 있어야 실루엣 점수 계산 가능
      non_noise_data = data[labels != -1]
      non_noise_labels = labels[labels != -1]
      score = silhouette_score(non_noise_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']], non_noise_labels)
      dbscan_scores[(eps, min_samples)] = score
      silhouette_scores["DBSCAN"][(eps, min_samples)] = score
      print(f"eps={eps}, min_samples={min_samples}, Silhouette Score={score:.4f}")
    else:
      print(f"eps={eps}, min_samples={min_samples}, insufficient clusters (only noise or 1 cluster).")

# 실루엣 점수가 가장 높은 K를 저장
best_k = {}

# KMeans, Hierarchical Clustering, GMM 처리
for method in ["KMeans", "Hierarchical Clustering", "GMM"]:
  scores = silhouette_scores[method]
  best_k[method] = max(scores, key=scores.get)

# DBSCAN 처리
if silhouette_scores["DBSCAN"]:
  best_dbscan_params = max(silhouette_scores["DBSCAN"], key=silhouette_scores["DBSCAN"].get)
  best_k["DBSCAN"] = best_dbscan_params

# 결과 출력
for method, k in best_k.items():
  if method == "DBSCAN":
    print(f"Best params for {method}: eps={k[0]}, min_samples={k[1]} with Silhouette Score: {silhouette_scores[method][k]}")
  else:
    print(f"Best k for {method}: {k} with Silhouette Score: {silhouette_scores[method][k]}")


# 엘보우 방법 시각화
plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), wcss, marker='o')
plt.title('Elbow Method for KMeans')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid()
plt.show()

# 실루엣 점수 시각화
plt.figure(figsize=(10, 6))

# KMeans, Hierarchical, GMM 점수 시각화
for method, scores in silhouette_scores.items():
    if method != "DBSCAN":
        plt.plot(scores.keys(), scores.values(), label=method, marker='o')

# DBSCAN 점수 시각화
dbscan_x = [f"{eps}-{min_samples}" for eps, min_samples in silhouette_scores["DBSCAN"].keys()]
dbscan_y = silhouette_scores["DBSCAN"].values()
plt.plot(dbscan_x, dbscan_y, label="DBSCAN", marker='x', linestyle='--')

plt.title('Silhouette Scores for Different Clustering Methods')
plt.xlabel('Number of Clusters (k) or DBSCAN Params (eps-min_samples)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()

```

### 2. 결과 시각화

```python
# 최적 클러스터링 및 점수 확인

# 1. KMeans
kmeans = KMeans(n_clusters=best_k['KMeans'], random_state=42)
kmeans_labels = kmeans.fit_predict(data)
kmeans_score = silhouette_score(data, kmeans_labels)
cluster_centers = kmeans.cluster_centers_

# KMeans 결과 시각화
# KMeans 결과 시각화 (2D Scatter Plot)
plt.figure(figsize=(8, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], 
            c=kmeans_labels, cmap='viridis', s=50, alpha=0.7, label='Cluster Data')
plt.scatter(cluster_centers[:, 1], cluster_centers[:, 2], 
            c='red', marker='X', s=200, label='Cluster Centers')
plt.title('KMeans Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'], 
                      c=kmeans_labels, cmap='viridis', s=50)
ax.set_title('3D KMeans Clustering')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.colorbar(scatter, label='Cluster Label')
plt.show()

print(f"KMeans 실루엣 스코어 = {kmeans_score}")


# 2. 계층적 군집화
hc = AgglomerativeClustering(n_clusters=best_k['Hierarchical Clustering'])
hc_labels = hc.fit_predict(data)
hc_score = silhouette_score(data, hc_labels)
print(f"계층적 군집화 실루엣 스코어 = {hc_score}")

# 3. DBSCAN
dbscan = DBSCAN(eps=best_dbscan_params[0], min_samples=best_dbscan_params[1])
dbscan_labels = dbscan.fit_predict(data)

non_noise_data = data[dbscan_labels != -1]
non_noise_labels = dbscan_labels[dbscan_labels != -1]
dbscan_score = silhouette_score(non_noise_data, non_noise_labels)
print(f"DBSCAN 실루엣 스코어 (without noise) = {dbscan_score}")

# DBSCAN 데이터 손실 비율 계산
total_data_points = len(data)  # 전체 데이터 포인트 수
noise_points = len(data[dbscan_labels == -1])  # 노이즈로 분류된 데이터 포인트 수
loss_ratio = noise_points / total_data_points  # 손실 비율 계산

# 손실 비율 출력
print(f"DBSCAN 데이터 손실 비율: {loss_ratio:.2%}")

# 4. GMM
gmm = GaussianMixture(n_components=best_k['GMM'], random_state=42)
gmm_labels = gmm.fit_predict(data)
gmm_score = silhouette_score(data, gmm_labels)
print(f"GMM 실루엣 스코어 = {gmm_score}")
```