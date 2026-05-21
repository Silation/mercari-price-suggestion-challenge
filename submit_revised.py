import os
# 하부 C/C++ 라이브러리의 OpenMP 멀티 스레딩을 1개로 강제 제한하여 코어 경합 방지
os.environ['OMP_NUM_THREADS'] = '1'

import time
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

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

def fit_predict(x_data, y_train, lr, batch_base, epochs, hidden_size) -> np.ndarray:
    X_train, X_test = x_data
    
    # 🌟 [복구된 핵심] TF 1.x 환경에 맞춘 스레드 제한 및 세션(Session) 격리
    # 각 스레드(코어)가 자신만의 독립된 그래프와 세션을 가지도록 설정
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, 
        use_per_session_threads=1, 
        inter_op_parallelism_threads=1
    )
    
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        ks.backend.set_session(sess)
        
        model_in = Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = Dense(hidden_size, activation='relu')(model_in)
        out = Dense(64, activation='relu')(out)
        out = Dense(64, activation='relu')(out)
        out = Dense(1)(out)
        model = Model(inputs=model_in, outputs=out)
        
        # 구형 Keras 호환을 위해 learning_rate 대신 lr 파라미터 사용
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
        
        for i in range(epochs):
            batch_s = batch_base * (2**i)
            model.fit(x=X_train, y=y_train, batch_size=batch_s, epochs=1, verbose=0)
            
        return model.predict(X_test)[:, 0]

# 병렬 매핑을 위한 Wrapper 함수
def fit_predict_wrapper(args):
    x_data, y_train, lr, batch_base, epochs, hidden_size = args
    return fit_predict(x_data, y_train, lr, batch_base, epochs, hidden_size)
    
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

    print(" Loading train.tsv (100% Full Data)...")
    train = pd.read_table('train.tsv')
    train = train[train['price'] > 0].reset_index(drop=True)
    
    print(" Loading test_stg2.tsv (Real Test Data)...")
    test = pd.read_table('test_stg2.tsv')
    
    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    
    print(" Extracting TF-IDF features...")
    X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
    X_test = vectorizer.transform(preprocess(test)).astype(np.float32)
    
    del train 

    Xb_train, Xb_test = [x.astype(bool).astype(np.float32) for x in [X_train, X_test]]
    
    xs = [[Xb_train, Xb_test], [X_train, X_test]]

    # ---------------------------------------------------------
    # [Step 1 & 2] Training ALL Models in Parallel (4 Cores)
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" [Phase 4-5] Training All Models Simultaneously (Parallel Mode)")
    print("="*75)
    
    # 4개의 모델(Baseline 2개, Hetero 2개)의 하이퍼파라미터 작업을 하나의 리스트로 통합
    tasks = [
        # Group A (원본 Baseline - 강력한 기준점 유지)
        (xs[0], y_train, 3e-3, 2048, 3, 192),
        (xs[1], y_train, 3e-3, 2048, 3, 192),
        
        # Group B (새로 찾은 Dual Hetero 최적 조합 - 다양성 극대화)
        (xs[0], y_train, 0.0015246, 1024, 3, 256),  # 3에폭 깊은 학습
        (xs[1], y_train, 0.0026261, 2048, 2, 128)   # 2에폭 얕은 학습
    ]
    
    print("   ▶ Spawning 4 processes... (Mapping 4 Tasks to 4 Physical Cores)")
    # 4개의 스레드를 띄워 4개의 작업을 일제히 시작
    with ThreadPool(processes=4) as pool:
        results = pool.map(fit_predict_wrapper, tasks)
        
    # 결과물을 리스트에서 분리하여 수합
    preds_baseline = results[0:2]
    preds_hetero = results[2:4]
    
    print("   ▶ All parallel models successfully trained and predicted!")

    # ---------------------------------------------------------
    # [Step 3] OR Strategy: 황금 가중치(Golden Weights) 적용
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" [Phase 6] OR Strategy: Applying Pre-calculated Golden Weights")
    print("="*75)
    
    shipping_flags = test['shipping'].values
    matrix_all = np.column_stack(preds_baseline + preds_hetero) 
    
    print("   ▶ Applying SciPy Golden Weights to Test Data...")
    w_golden_0 = np.array([0.178899, 0.296864, 0.223596, 0.300641])
    w_golden_1 = np.array([0.214568, 0.282913, 0.271665, 0.230855])
    
    preds_final = np.zeros(len(test))
    preds_final[shipping_flags == 0] = np.dot(matrix_all[shipping_flags == 0], w_golden_0)
    preds_final[shipping_flags == 1] = np.dot(matrix_all[shipping_flags == 1], w_golden_1)
    
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
    
    submission.to_csv('submission.csv', index=False)
    print(" ✅ submission.csv successfully generated! Ready to Submit!")
    print("="*75)

if __name__ == '__main__':
    main()