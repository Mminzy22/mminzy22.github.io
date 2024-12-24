---
layout: post
title: "Machine Learning 39: 과제: 고객 세분화 분석"
date: 2024-12-24
categories: [Machine Learning]
tag: []
---

### 과제 1번: 비지도학습

#### 주제: 고객 세분화 분석
- [고객 데이터셋]({{ site.baseurl }}/assets/downloads/Mall_Customers.csv)을 사용하여 비슷한 행동을 보이는 고객 그룹을 식별합니다.

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
