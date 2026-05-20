import os
import time
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

# 구형 Keras 완벽 호환
import tensorflow as tf
import keras as ks
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] Elapsed Time: {time.time() - t0:.0f} seconds\n')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    return df[['name', 'text', 'shipping', 'item_condition_id']]

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train, lr, batch_base, epochs, hidden_size) -> np.ndarray:
    X_train, X_test = xs
    
    model_in = Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
    out = Dense(hidden_size, activation='relu')(model_in)
    out = Dense(64, activation='relu')(out)
    out = Dense(64, activation='relu')(out)
    out = Dense(1)(out)
    model = Model(inputs=model_in, outputs=out)
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
    
    for i in range(epochs):
        batch_s = batch_base * (2**i)
        model.fit(x=X_train, y=y_train, batch_size=batch_s, epochs=1, verbose=0)
        
    return model.predict(X_test)[:, 0]
    
def main():
    print("="*75)
    print(" [Phase 1-3] Data Loading and Processing (100% Train & Stage 2 Test) ")
    print("="*75)
    vectorizer = make_union(
        on_field('name', Tfidf(max_features=100000, token_pattern='\w+')),
        on_field('text', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))),
        on_field(['shipping', 'item_condition_id'], FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=4)
    y_scaler = StandardScaler()

    # 1. Train 데이터를 쪼개지 않고 100% 모두 가져옵니다.
    print(" Loading train.tsv (100% Full Data)...")
    train = pd.read_table('train.tsv')
    train = train[train['price'] > 0].reset_index(drop=True)
    
    # 2. 정답이 없는 진짜 수능 시험지(Test Stage 2)를 가져옵니다.
    print(" Loading test_stg2.tsv (Real Test Data)...")
    test = pd.read_table('test_stg2.tsv')
    
    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    
    print(" Extracting TF-IDF features...")
    X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
    X_test = vectorizer.transform(preprocess(test)).astype(np.float32)
    
    # 메모리 확보를 위해 학습이 끝난 텍스트 원본 데이터 삭제
    del train 

    Xb_train, Xb_test = [x.astype(bool).astype(np.float32) for x in [X_train, X_test]]
    
    # 시간에 쫓기지 않기 위해 중복(* 2)을 제거한 2세트(총 4개 모델) 설정
    xs = [[Xb_train, Xb_test], [X_train, X_test]]

    # ---------------------------------------------------------
    # [Step 1] Baseline Group 학습 (140만 개 전체 데이터)
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" [Phase 4] Training Baseline Models (100% Data, Sequential Mode)")
    print("="*75)
    preds_baseline = []
    for idx, x_data in enumerate(xs):
        print(f"   - Training Baseline Model {idx+1}/{len(xs)}...")
        pred = fit_predict(x_data, y_train, lr=3e-3, batch_base=2048, epochs=3, hidden_size=192)
        preds_baseline.append(pred)

    # ---------------------------------------------------------
    # [Step 2] Heterogeneous Group 학습 (140만 개 전체 데이터)
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" [Phase 5] Training Heterogeneous Models (100% Data, Sequential Mode)")
    print("="*75)
    preds_hetero = []
    for idx, x_data in enumerate(xs):
        print(f"   - Training Hetero Model {idx+1}/{len(xs)}...")
        pred = fit_predict(x_data, y_train, lr=0.00161, batch_base=1024, epochs=1, hidden_size=256)
        preds_hetero.append(pred)

    # ---------------------------------------------------------
    # [Step 3] OR Strategy: 황금 가중치(Golden Weights) 적용
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" [Phase 6] OR Strategy: Applying Pre-calculated Golden Weights")
    print("="*75)
    
    shipping_flags = test['shipping'].values
    matrix_all = np.column_stack(preds_baseline + preds_hetero) # 총 4개의 예측 결과(열)
    
    # 🌟 [매우 중요] 이전 모의고사(Validation) 단계에서 터미널에 출력되었던 
    # w_all0 와 w_all1 의 실제 숫자 4개를 아래 배열에 적어주세요!
    # (현재는 예시로 단순 1/N 평균값을 넣어두었습니다.)
    print("   ▶ Applying SciPy Golden Weights to Test Data...")
    w_golden_0 = np.array([0.241525, 0.321700, 0.172541, 0.264233]) # 배송비 0일 때의 가중치 4개
    w_golden_1 = np.array([0.288720, 0.377159, 0.162786, 0.171336]) # 배송비 1일 때의 가중치 4개
    
    preds_final = np.zeros(len(test))
    preds_final[shipping_flags == 0] = np.dot(matrix_all[shipping_flags == 0], w_golden_0)
    preds_final[shipping_flags == 1] = np.dot(matrix_all[shipping_flags == 1], w_golden_1)
    
    # 역변환하여 실제 달러($) 가격으로 복원
    final_price = np.expm1(y_scaler.inverse_transform(preds_final.reshape(-1, 1))[:, 0])

    # ---------------------------------------------------------
    # [Step 4] 최종 제출 파일 생성
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" [Phase 7] Generating Submission File ")
    print("="*75)
    submission = pd.DataFrame({
        'test_id': test['test_id'],
        'price': final_price
    })
    
    # 캐글 서버의 /kaggle/working/ 폴더에 저장됨
    submission.to_csv('submission_stg2.csv', index=False)
    print(" ✅ submission_stg2.csv successfully generated! Ready to Submit!")
    print("="*75)

if __name__ == '__main__':
    main()