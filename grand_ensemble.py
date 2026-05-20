import os
os.environ['OMP_NUM_THREADS'] = '1'

from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

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
import gurobipy as gp
from gurobipy import GRB

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
    
    model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(hidden_size, activation='relu')(model_in)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(learning_rate=lr))
    
    for i in range(epochs):
        batch_s = batch_base * (2**i)
        model.fit(x=X_train, y=y_train, batch_size=batch_s, epochs=1, verbose=0)
    return model.predict(X_test)[:, 0]

def optimize_weights_gurobi(preds_matrix, y_true):
    N_models = preds_matrix.shape[1]
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Ensemble_Weight_Optimization", env=env)
    
    w = m.addVars(N_models, lb=0.0, ub=1.0, name="w")
    m.addConstr(gp.quicksum(w[i] for i in range(N_models)) == 1.0, "SumToOne")
    
    PTP = np.dot(preds_matrix.T, preds_matrix) 
    PTy = np.dot(preds_matrix.T, y_true)       
    
    obj = gp.QuadExpr()
    for i in range(N_models):
        for j in range(N_models):
            obj += PTP[i, j] * w[i] * w[j]
    for i in range(N_models):
        obj -= 2.0 * PTy[i] * w[i]
        
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        return np.array([w[i].X for i in range(N_models)])
    return np.ones(N_models) / N_models

def main():
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("="*60)
    print(" [Phase 1-3] Data Loading and Processing ")
    print("="*60)
    vectorizer = make_union(
        on_field('name', Tfidf(max_features=100000, token_pattern='\w+')),
        on_field('text', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))),
        on_field(['shipping', 'item_condition_id'], FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=4)
    y_scaler = StandardScaler()

    train_raw = pd.read_table('input/train.tsv')
    train_raw = train_raw[train_raw['price'] > 0].reset_index(drop=True)
    
    # 🌟 빠른 검증을 위해 반드시 샘플링을 켜고 먼저 테스트해보세요!
    # train_raw = train_raw.sample(frac=0.05, random_state=42).reset_index(drop=True)

    cv = KFold(n_splits=20, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train_raw))
    train = train_raw.iloc[train_ids].copy()
    valid = train_raw.iloc[valid_ids].copy()
    del train_raw
    
    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    y_valid_scaled = y_scaler.transform(np.log1p(valid['price'].values.reshape(-1, 1)))[:, 0]
    
    print(" Extracting TF-IDF features...")
    X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
    del train
    X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)

    Xb_train, Xb_valid = [x.astype(bool).astype(np.float32) for x in [X_train, X_valid]]
    xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2

    # ---------------------------------------------------------
    # 1. 모델 그룹 A (기존 AS-IS 초강력 베이스라인 4개) 학습
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" [Step 1] Training Group A (Original AS-IS Models)")
    print("="*60)
    with ThreadPool(processes=4) as pool:
        preds_A = pool.map(partial(fit_predict, y_train=y_train, lr=3e-3, batch_base=2048, epochs=3, hidden_size=192), xs)

    # ---------------------------------------------------------
    # 2. 모델 그룹 B (Optuna가 찾은 이종 다양성 모델 4개) 학습
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" [Step 2] Training Group B (Optuna TO-BE Models for Diversity)")
    print("="*60)
    with ThreadPool(processes=4) as pool:
        preds_B = pool.map(partial(fit_predict, y_train=y_train, lr=0.00161, batch_base=1024, epochs=1, hidden_size=256), xs)

    # ---------------------------------------------------------
    # 3. Gurobi 최적화 (8개 모델 대통합)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" [Step 3] Gurobi Optimization on ALL 8 Models")
    print("="*60)
    shipping_flags = valid['shipping'].values
    
    # 평가 1: 기존 그룹 A만의 최적화 (대조군 최고 기록)
    matrix_A = np.column_stack(preds_A)
    w_A0 = optimize_weights_gurobi(matrix_A[shipping_flags == 0], y_valid_scaled[shipping_flags == 0])
    w_A1 = optimize_weights_gurobi(matrix_A[shipping_flags == 1], y_valid_scaled[shipping_flags == 1])
    
    preds_A_seg = np.zeros(len(valid))
    preds_A_seg[shipping_flags == 0] = np.dot(matrix_A[shipping_flags == 0], w_A0)
    preds_A_seg[shipping_flags == 1] = np.dot(matrix_A[shipping_flags == 1], w_A1)
    score_A_best = np.sqrt(mean_squared_log_error(valid['price'], np.expm1(y_scaler.inverse_transform(preds_A_seg.reshape(-1, 1))[:, 0])))

    # 평가 2: 8개 모델 대통합 최적화 (신기록 도전)
    matrix_ALL = np.column_stack(preds_A + preds_B) # 8개 열의 거대 행렬
    w_ALL0 = optimize_weights_gurobi(matrix_ALL[shipping_flags == 0], y_valid_scaled[shipping_flags == 0])
    w_ALL1 = optimize_weights_gurobi(matrix_ALL[shipping_flags == 1], y_valid_scaled[shipping_flags == 1])
    
    preds_ALL_seg = np.zeros(len(valid))
    preds_ALL_seg[shipping_flags == 0] = np.dot(matrix_ALL[shipping_flags == 0], w_ALL0)
    preds_ALL_seg[shipping_flags == 1] = np.dot(matrix_ALL[shipping_flags == 1], w_ALL1)
    score_ALL_best = np.sqrt(mean_squared_log_error(valid['price'], np.expm1(y_scaler.inverse_transform(preds_ALL_seg.reshape(-1, 1))[:, 0])))

    # ---------------------------------------------------------
    # 최종 리포트 출력
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" 🏆 [GRAND FINALE : OR 최적화 한계 돌파 리포트] 🏆")
    print("="*75)
    print(f" 1. 기존 베이스라인 최고 기록 (A모델 4개 + QP)    : {score_A_best:.5f}")
    print(f" 2. 이종 아키텍처 대통합 기록 (8개 모델 + QP)      : {score_ALL_best:.5f} 🎯")
    print("-" * 75)
    print(f" 💡 성능 한계 돌파 (추가 오차 감소폭)              : -{score_A_best - score_ALL_best:.5f}")
    print("="*75)
    print("\n [Gurobi 가중치 분석 (배송비=0 기준)]")
    print(f" - 강력한 기존 모델 (A) 가중치 합: {np.sum(w_ALL0[:4]):.3f}")
    print(f" - 약하지만 새로운 관점의 모델 (B) 가중치 합: {np.sum(w_ALL0[4:]):.3f}")

if __name__ == '__main__':
    main()