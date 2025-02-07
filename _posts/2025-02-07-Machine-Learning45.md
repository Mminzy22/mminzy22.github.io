---
title: "Hugging Face의 Transformers 라이브러리로 NLP 모델 활용하기"
author: mminzy22
date: 2025-02-07 20:00:00 +0900
categories: [Machine Learning]
tags: [Bootcamp, Python, Machine Learning, Hugging Face, Transformers, Deep Learning, AI, TIL]
description: "Hugging Face의 Transformers 라이브러리를 활용해 BERT, GPT-2, RoBERTa 등 다양한 NLP 모델을 쉽게 사용하는 방법"
pin: false
math: true
---


## 1. Transformers 라이브러리란?

Hugging Face의 **`Transformers`** 라이브러리는 다양한 자연어 처리(NLP) 모델을 쉽게 사용할 수 있도록 지원하는 오픈소스 라이브러리입니다. 이 라이브러리를 활용하면 최신 NLP 모델들을 간편하게 불러와 텍스트 생성, 감성 분석, 번역 등의 작업을 수행할 수 있습니다.

## 2. 실습 전 준비하기

본격적인 실습을 진행하기 전에 필요한 설정을 먼저 완료해야 합니다.

### 가상 환경 활성화

```bash
conda activate 환경이름
```

### Transformers 라이브러리 버전 낮추기

```bash
pip install transformers==4.37.0
```

### Python 경고 메시지 숨기기

```python
import warnings
warnings.filterwarnings('ignore')
```

## 3. 다양한 NLP 모델 활용하기

### 1. GPT-2로 텍스트 생성하기

`GPT-2`는 OpenAI에서 개발한 언어 생성 모델로, 문장을 생성하거나 이어지는 텍스트를 예측하는 데 뛰어난 성능을 발휘합니다.

```python
from transformers import pipeline

# GPT-2 기반 텍스트 생성 파이프라인 로드
generator = pipeline("text-generation", model="gpt2")

# 텍스트 생성
result = generator("Once upon a time", max_length=50, num_return_sequences=1)
print(result)
```

### 2. 간단한 감성어 분석

감성 분석(Sentiment Analysis)은 텍스트가 긍정적인지, 부정적인지를 판단하는 작업입니다.

```python
from transformers import pipeline

# 감정 분석 파이프라인 로드
sentiment_analysis = pipeline("sentiment-analysis")
result = sentiment_analysis("I love using Hugging Face!")
print(result)
```

### 3. RoBERTa를 활용한 감정 분석

`RoBERTa`는 `BERT` 모델을 최적화한 버전으로, 텍스트 분류 및 감성 분석에서 뛰어난 성능을 보입니다.

```python
from transformers import pipeline

# RoBERTa 기반 감정 분석 파이프라인 로드
classifier = pipeline("sentiment-analysis", model="roberta-base")

# 감정 분석 실행
result = classifier("This product is amazing!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

## 4. BERT 바로 사용이 어려운 이유

`BERT`는 사전 학습된 언어 모델로, 기본적으로 텍스트 분류와 같은 작업을 수행하려면 **파인튜닝(Fine-tuning)**이 필요합니다. BERT의 기본 모델은 일반적인 언어 모델링만 학습되어 있기 때문에, 추가적인 훈련이 없으면 적절한 예측을 하지 못할 가능성이 큽니다.

## 5. 실습 진행하기

### 1. Word2Vec 기법 사용하기

Word2Vec은 단어 간의 유사도를 계산하는 대표적인 임베딩 기법입니다.

```python
# gensim 설치하기
pip install gensim

# 필요한 것 불러오기
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from scipy.spatial.distance import cosine

# 예시 문장 입력하기
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "I love playing with my pet dog",
    "The dog barks at the stranger",
    "The cat sleeps on the sofa",
]

# 문장을 단어 단위로 변환
processed = [simple_preprocess(sentence) for sentence in sentences]

# Word2Vec 모델 학습
model = Word2Vec(sentences=processed, vector_size=5, window=5, min_count=1, sg=0)

# 단어 임베딩 벡터 확인
dog = model.wv['dog']
cat = model.wv['cat']

# 두 단어 간의 유사도 계산
sim = 1 - cosine(dog, cat)
print(f"Cosine similarity between 'dog' and 'cat': {sim:.4f}")
```

### 2. BERT 기반의 문장 임베딩 비교하기

BERT를 활용해 문장 간의 의미적 유사도를 비교할 수 있습니다.

```python
from transformers import BertModel, BertTokenizer
import torch
from scipy.spatial.distance import cosine

# BERT 모델 로드
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 비교할 문장 입력
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "A fast brown fox leaps over a sleepy dog"
]

# 문장 토큰화 및 입력 텐서 생성
input1 = tokenizer(sentences[0], return_tensors='pt')
input2 = tokenizer(sentences[1], return_tensors='pt')

# 모델을 사용하여 문장 임베딩 생성
with torch.no_grad():
    output1 = model(**input1)
    output2 = model(**input2)

# 평균 풀링을 통한 문장 벡터 생성
embedding1 = output1.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
embedding2 = output2.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# 코사인 유사도 계산
similarity = 1 - cosine(embedding1, embedding2)
print(f"Cosine similarity between the two sentences: {similarity:.4f}")
```

### 3. 번역 모델 활용하기

#### M2M100 모델을 이용한 번역

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# 모델 및 토크나이저 로드
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# 번역할 문장
sentence = "The quick brown fox jumps over the lazy dog"

# 입력 문장을 토큰화
tokenizer.src_lang = "en"
encoded_sentence = tokenizer(sentence, return_tensors="pt")

# 번역 실행
generated_tokens = model.generate(**encoded_sentence)
translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

print(f"Translated text: {translated_text}")
```


이처럼 **Hugging Face의 Transformers 라이브러리**를 활용하면 최신 NLP 모델을 쉽게 사용할 수 있습니다. 각 모델은 특정 작업에 최적화되어 있으며, **텍스트 생성, 감성 분석, 번역, 문장 임베딩 등 다양한 자연어 처리 기능**을 효율적으로 수행할 수 있습니다.

