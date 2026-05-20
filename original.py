import os
# 전역 환경에서 CPU 스레드 최적화 설정을 걸어둡니다.
os.environ['OMP_NUM_THREADS'] = '1'

from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

# 🚀 [개선 포인트] 과거 호환성 레이어(tf1)를 완전히 제거하고 순수 TF2/Keras3 기반으로 전환
import tensorflow as tf
import keras as ks

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] 완료 소요시간: {time.time() - t0:.0f} 초\n')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    return df[['name', 'text', 'shipping', 'item_condition_id']]

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

# 🚀 [개선 포인트] 복잡한 tf.Session() 및 ks.backend.set_session 제거
def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
    
    # Keras 3는 이 블록이 호출될 때마다 독립된 메모리 공간에 모델 그래프를 자동으로 격리 생성합니다.
    model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(192, activation='relu')(model_in)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    
    # 최신 Keras 규격에 맞게 인자명을 lr에서 learning_rate로 변경
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(learning_rate=3e-3))
    
    print(f"   ▶ [모델 학습 시작] 데이터 크기: {X_train.shape} ... (총 3 Epoch 진행)")
    for i in range(3):
        batch_s = 2**(11 + i)
        model.fit(x=X_train, y=y_train, batch_size=batch_s, epochs=1, verbose=0)
        print(f"      - Epoch {i+1}/3 완료 (Batch Size: {batch_s})")
        
    print("   ▶ [모델 학습 종료] 검증 데이터 예측값 추출 중...\n")
    return model.predict(X_test)[:, 0]

def main():
    # 🚀 [개선 포인트] TF2 방식의 글로벌 CPU 코어 선점 제한 (4개 모델이 사이좋게 1코어씩 나눠 쓰도록 설정)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("="*60)
    print(" [Phase 1] 데이터 파이프라인 및 벡터화 도구 세팅")
    print("="*60)
    vectorizer = make_union(
        on_field('name', Tfidf(max_features=100000, token_pattern='\w+')),
        on_field('text', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))),
        on_field(['shipping', 'item_condition_id'],
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=4)
    y_scaler = StandardScaler()

    print()
    print("="*60)
    print(" [Phase 2] Train 데이터 로드 및 전처리 진행")
    print("="*60)
    with timer('Phase 2 전체'):
        train = pd.read_table('input/train.tsv')
        train = train[train['price'] > 0].reset_index(drop=True)
        print(f"원본 데이터 로드 완료: 총 {len(train)}개 행")
        
        cv = KFold(n_splits=20, shuffle=True, random_state=42)
        train_ids, valid_ids = next(cv.split(train))
        train, valid = train.iloc[train_ids], train.iloc[valid_ids]
        print(f"데이터 분할 완료 -> 학습용(Train): {len(train)}개 / 검증용(Valid): {len(valid)}개")
        
        y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
        
        print("텍스트 데이터를 TF-IDF 숫자로 변환 중 (이 작업은 시간이 조금 걸립니다)...")
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        print(f"Train 피처 변환 완료: 차원 크기 = {X_train.shape}")
        del train

    print()
    print("="*60)
    print(" [Phase 3] Valid 데이터 전처리 진행")
    print("="*60)
    with timer('Phase 3 전체'):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
        print(f"Valid 피처 변환 완료: 차원 크기 = {X_valid.shape}")

    print()
    print("="*60)
    print(" [Phase 4] 멀티프로세싱 기반 4개 모델 병렬 학습 시작")
    print("="*60)
    with ThreadPool(processes=4) as pool:
        print("1. 오리지널 TF-IDF 데이터를 0과 1로 변환한 '바이너리 데이터셋' 복제 중...")
        Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
        xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
        
        print("2. 4개의 독립된 신경망 모델 학습을 동시에 출발시킵니다!\n")
        preds_list = pool.map(partial(fit_predict, y_train=y_train), xs)
        print("모든 모델 학습 및 예측 완료!")

    print()
    print("="*60)
    print(" [Phase 5] 앙상블(Ensemble) 및 최종 RMSLE 평가")
    print("="*60)
    y_pred = np.mean(preds_list, axis=0)
    
    y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
    
    final_score = np.sqrt(mean_squared_log_error(valid['price'], y_pred))
    print(f"🎯 최종 Valid RMSLE 점수: {final_score:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()