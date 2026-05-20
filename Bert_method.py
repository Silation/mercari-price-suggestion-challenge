import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from transformers import AutoTokenizer, TFAutoModel

# ---------------------------------------------------------
# 1. 환경 설정 및 유틸리티
# ---------------------------------------------------------
MAX_LEN = 128  # 텍스트 최대 토큰 길이 (GPU 메모리에 맞춰 조절 가능)
BATCH_SIZE = 256 # TF-IDF 떄보다 배치 사이즈를 줄여야 GPU OOM을 피할 수 있음

def preprocess_text(df: pd.DataFrame) -> pd.Series:
    """기존 코드와 동일하게 텍스트 필드를 하나로 병합"""
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    text = (df['item_description'].fillna('') + ' ' + 
            df['name'] + ' ' + 
            df['category_name'].fillna(''))
    return text

# ---------------------------------------------------------
# 2. 텍스트 토큰화 및 데이터 제너레이터 구성
# ---------------------------------------------------------
def prepare_bert_inputs(texts, tokenizer, max_len=MAX_LEN):
    """텍스트를 BERT가 이해할 수 있는 토큰(input_ids, attention_mask)으로 변환"""
    encoded = tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )
    return encoded['input_ids'], encoded['attention_mask']

# ---------------------------------------------------------
# 3. 모델 아키텍처 정의 (Two-Stream)
# ---------------------------------------------------------
def build_bert_mlp_model(meta_dim, lr=3e-3, hidden_size=256):
    # [Stream 1] 텍스트 입력층 (BERT용)
    input_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')
    
    # [Stream 2] 메타데이터 입력층 (배송비, 상품상태 등)
    meta_inputs = tf.keras.Input(shape=(meta_dim,), dtype=tf.float32, name='meta_inputs')

    # BERT 모델 로드 (가벼운 DistilBERT 사용)
    bert_base = TFAutoModel.from_pretrained('distilbert-base-uncased')
    
    # 🌟 핵심 트릭: BERT 자체의 가중치는 업데이트하지 않음 (Freeze)
    # TF-IDF를 쓰던 기존 방식처럼 BERT를 '고밀도 특징 추출기'로만 사용해 MLP의 안정성 확보
    bert_base.trainable = False 

    # BERT 통과 후 [CLS] 토큰 위치의 임베딩(문장 전체의 문맥 정보) 추출
    bert_outputs = bert_base(input_ids, attention_mask=attention_mask)
    cls_token = bert_outputs[0][:, 0, :] # shape: (batch_size, 768)

    # BERT 임베딩(768차원)과 메타데이터 결합
    concat = tf.keras.layers.Concatenate()([cls_token, meta_inputs])

    # 🌟 기존 코드의 MLP 뼈대 완벽 복원
    out = tf.keras.layers.Dense(hidden_size, activation='relu')(concat)
    out = tf.keras.layers.Dense(64, activation='relu')(out)
    out = tf.keras.layers.Dense(64, activation='relu')(out)
    out = tf.keras.layers.Dense(1)(out) # 선형 활성화로 최종 가격(로그 스케일) 예측

    model = tf.keras.Model(inputs=[input_ids, attention_mask, meta_inputs], outputs=out)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    
    return model

# ---------------------------------------------------------
# 4. 메인 파이프라인
# ---------------------------------------------------------
def main():
    print(" [1] Data Loading & Preprocessing...")
    train_raw = pd.read_table('input/train.tsv')
    train_raw = train_raw[train_raw['price'] > 0].reset_index(drop=True)
    
    # Train / Valid 분할
    cv = KFold(n_splits=20, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train_raw))
    train = train_raw.iloc[train_ids].copy()
    valid = train_raw.iloc[valid_ids].copy()
    del train_raw
    
    # 타겟 스케일링 (기존 로직 동일)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    y_valid = y_scaler.transform(np.log1p(valid['price'].values.reshape(-1, 1)))
    
    print(" [2] Preparing Meta Features (DictVectorizer)...")
    # 메타데이터 추출 (TF-IDF 대신 DictVectorizer만 사용)
    dv = DictVectorizer()
    meta_train_dict = train[['shipping', 'item_condition_id']].to_dict(orient='records')
    meta_valid_dict = valid[['shipping', 'item_condition_id']].to_dict(orient='records')
    
    meta_train = dv.fit_transform(meta_train_dict).toarray().astype(np.float32)
    meta_valid = dv.transform(meta_valid_dict).toarray().astype(np.float32)

    print(" [3] Tokenizing Text Data with Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_text = preprocess_text(train)
    valid_text = preprocess_text(valid)
    
    train_input_ids, train_attention_mask = prepare_bert_inputs(train_text, tokenizer)
    valid_input_ids, valid_attention_mask = prepare_bert_inputs(valid_text, tokenizer)

    print(" [4] Building and Training Model...")
    model = build_bert_mlp_model(meta_dim=meta_train.shape[1], lr=3e-3, hidden_size=256)
    model.summary() # 모델 구조 확인

    # 동적 배치 스케줄링 적용 (기존 로직 계승, 단 GPU 메모리에 맞게 기본값 하향)
    epochs = 3
    batch_base = 256
    
    for i in range(epochs):
        batch_s = batch_base * (2**i)
        print(f"\n --- Epoch {i+1} (Batch Size: {batch_s}) ---")
        model.fit(
            x=[train_input_ids, train_attention_mask, meta_train],
            y=y_train,
            batch_size=batch_s,
            epochs=1,
            verbose=1 # 학습 과정을 보기 위해 1로 설정
        )

    print("\n [5] Predicting & Evaluating...")
    preds = model.predict([valid_input_ids, valid_attention_mask, meta_valid], batch_size=1024)[:, 0]
    
    # 타겟 원상 복구 및 RMSLE 계산
    preds_inv = np.expm1(y_scaler.inverse_transform(preds.reshape(-1, 1))[:, 0])
    score = np.sqrt(mean_squared_log_error(valid['price'], preds_inv))
    
    print("="*60)
    print(f" 🏆 [Validation RMSLE] : {score:.5f}")
    print("="*60)

if __name__ == '__main__':
    main()