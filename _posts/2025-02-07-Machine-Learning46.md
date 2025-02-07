---
title: "Hugging Face 모델의 사전 학습과 파인 튜닝"
author: mminzy22
date: 2025-02-07 20:30:00 +0900
categories: [Machine Learning]
tags: [Bootcamp, Python, Machine Learning, Hugging Face, Transformers, Deep Learning, AI, TIL]
description: "Hugging Face 모델의 사전 학습(Pre-training)과 파인 튜닝(Fine-tuning)의 개념을 배우고, IMDb 데이터셋을 활용해 BERT 모델을 파인 튜닝하는 방법"
pin: false
math: true
---


## 1. 사전 학습(Pre-training)

### 사전 학습이란?

**사전 학습(Pre-training)**은 대규모 텍스트 데이터셋을 사용해 모델이 일반적인 언어 이해 능력을 학습하는 과정입니다. 이 단계에서는 특정 작업(예: 번역, 감정 분석 등)에 맞춘 학습이 아닌, 언어의 패턴과 구조를 학습하는 것이 목적입니다.

### 사전 학습의 특징

1. **대규모 데이터셋 사용**
   - 인터넷에서 수집한 방대한 텍스트 데이터로 학습
   - 예: **`BERT`**는 수십억 개의 문장으로 학습됨

2. **일반적인 언어 이해**
   - 단어 의미, 문장 구조, 문맥을 학습

3. **작업 비특화**
   - 특정 작업이 아닌, 언어의 전반적인 특징을 학습

### 사전 학습의 목적

사전 학습을 통해 모델은 다양한 텍스트에서 언어의 기본적인 규칙을 배우고, 특정 작업에 빠르게 적응할 수 있는 기반을 다집니다. Hugging Face에서 제공하는 대부분의 모델은 이 단계를 완료한 상태로 제공됩니다.

### 예시: BERT의 사전 학습

**BERT**는 두 가지 주요 학습 과정을 수행합니다.

1. **Masked Language Modeling (MLM)**
   - 문장의 일부 단어를 마스킹한 후 이를 예측하도록 학습
   - 문맥을 양방향으로 이해하는 능력 강화

2. **Next Sentence Prediction (NSP)**
   - 두 문장이 연속된 문장인지 예측하는 작업
   - 문장 간의 관계를 학습


## 2. 파인 튜닝(Fine-tuning)

### 파인 튜닝이란?

**파인 튜닝(Fine-tuning)**은 사전 학습된 모델을 특정 작업에 맞게 추가 학습하는 과정입니다. 예를 들어, **`BERT`** 모델을 감정 분석에 사용하려면, 기존 가중치를 유지하면서 감정 분석 작업에 맞게 모델을 조정합니다.

### 파인 튜닝의 특징

1. **작업 특화**
   - 특정 작업(예: 텍스트 분류, 번역, 질의 응답 등)에 맞춰 최적화

2. **사전 학습 가중치 활용**
   - 기존의 언어 이해 능력을 바탕으로 새로운 작업에 적응

3. **적은 데이터로도 가능**
   - 사전 학습 덕분에 적은 데이터로도 효과적인 학습 가능

### 파인 튜닝의 목적

사전 학습된 모델을 특정 작업에서 최상의 성능을 발휘하도록 조정하는 과정입니다. Hugging Face의 모델들은 대부분 이 과정을 거쳐 다양한 애플리케이션에 활용됩니다.


## 3. IMDb 데이터셋을 활용한 BERT 파인 튜닝 실습

### 1. 필요한 라이브러리 설치 및 임포트

```bash
pip install transformers datasets torch accelerate -U
```

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
```

### 2. IMDb 데이터셋 로드

```python
# IMDb 데이터셋 로드
dataset = load_dataset("imdb")

# 훈련 및 테스트 데이터셋 분리
train_dataset = dataset['train'].shuffle(seed=42).select(range(1000))  # 1000개 샘플
test_dataset = dataset['test'].shuffle(seed=42).select(range(500))  # 500개 샘플
```

### 3. 데이터 전처리 및 토크나이저 적용

```python
# BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
```

### 4. BERT 모델 로드 및 파인 튜닝

```python
# BERT 모델 로드
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_steps=10_000,
    save_total_limit=2,
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 모델 훈련
trainer.train()
trainer.evaluate()
```

### 5. 모델 평가 및 결과 확인

```python
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)  # 예측된 클래스
    labels = p.label_ids  # 실제 레이블
    acc = accuracy_score(labels, preds)  # 정확도 계산
    return {'accuracy': acc}

trainer.compute_metrics = compute_metrics

# 모델 평가 및 정확도 확인
eval_result = trainer.evaluate()
print(f"Accuracy: {eval_result['eval_accuracy']:.4f}")
```


## 4. 사전 학습 vs 파인 튜닝 요약

| 구분 | 사전 학습(Pre-training) | 파인 튜닝(Fine-tuning) |
|------|------------------|------------------|
| 목적 | 언어의 일반적인 특징 학습 | 특정 작업에 최적화 |
| 데이터 | 대규모 텍스트 데이터 | 특정 작업에 맞는 데이터 |
| 학습 방식 | 언어 모델링(MLM, NSP) | 특정 태스크에 대한 미세 조정 |
| Hugging Face 모델 | `bert-base-uncased` 등 | `bert-base-uncased` + 감정 분석 태스크 |


## 5. 결론

- **사전 학습(Pre-training)**: 일반적인 언어 패턴과 문법을 이해하도록 대규모 텍스트 데이터로 모델을 학습
- **파인 튜닝(Fine-tuning)**: 사전 학습된 모델을 특정 작업(예: 텍스트 분류, 번역 등)에 맞게 추가 학습

이 두 단계를 거쳐 최신 NLP 모델이 높은 성능을 발휘할 수 있습니다. 사전 학습된 모델을 활용하고, 파인 튜닝을 통해 특정 작업에 최적화하면 훨씬 더 빠르고 효율적으로 AI 모델을 만들 수 있습니다.

