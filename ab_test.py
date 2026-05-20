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

# 파라미터를 동적으로 받도록 수정된 모델 훈련 함수
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

# Gurobi 최적화 함수
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

# 평가(Evaluation)를 수행하는 공통 모듈
def evaluate_model(preds_list, valid_df, y_scaler, y_valid_scaled):
    preds_matrix = np.column_stack(preds_list)
    
    # 1. 1/N Simple Ensemble
    preds_mean = np.mean(preds_list, axis=0)
    preds_mean_inv = np.expm1(y_scaler.inverse_transform(preds_mean.reshape(-1, 1))[:, 0])
    score_baseline = np.sqrt(mean_squared_log_error(valid_df['price'], preds_mean_inv))
    
    # 2. Segmented QP Optimization
    shipping_flags = valid_df['shipping'].values
    weights_ship0 = optimize_weights_gurobi(preds_matrix[shipping_flags == 0], y_valid_scaled[shipping_flags == 0])
    weights_ship1 = optimize_weights_gurobi(preds_matrix[shipping_flags == 1], y_valid_scaled[shipping_flags == 1])
    
    preds_segmented = np.zeros(len(valid_df))
    preds_segmented[shipping_flags == 0] = np.dot(preds_matrix[shipping_flags == 0], weights_ship0)
    preds_segmented[shipping_flags == 1] = np.dot(preds_matrix[shipping_flags == 1], weights_ship1)
    
    preds_seg_inv = np.expm1(y_scaler.inverse_transform(preds_segmented.reshape(-1, 1))[:, 0])
    score_segmented_qp = np.sqrt(mean_squared_log_error(valid_df['price'], preds_seg_inv))
    
    return score_baseline, score_segmented_qp

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
    
    # 디버깅/빠른 실험을 원하시면 아래 주석을 풀고 사용하세요 (전체 데이터의 5%만 사용)
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

    # 데이터 준비 (np.bool 대신 파이썬 내장 bool 또는 np.bool_ 사용)
    Xb_train, Xb_valid = [x.astype(bool).astype(np.float32) for x in [X_train, X_valid]]
    xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2

    # ---------------------------------------------------------
    # [실험 A] 기존 베이스라인 설정 (AS-IS)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" [Experiment A] AS-IS : Original Baseline Parameters")
    print(" Parameters: LR=0.003, Batch=2048, Epochs=3, Hidden=192")
    print("="*60)
    with ThreadPool(processes=4) as pool:
        preds_list_A = pool.map(
            partial(fit_predict, y_train=y_train, lr=3e-3, batch_base=2048, epochs=3, hidden_size=192), xs
        )
    score_A_base, score_A_qp = evaluate_model(preds_list_A, valid, y_scaler, y_valid_scaled)

    # ---------------------------------------------------------
    # [실험 B] Optuna + 베이지안 최적화 세팅 (TO-BE)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" [Experiment B] TO-BE : Optimized Parameters (Optuna)")
    print(" Parameters: LR=0.00161, Batch=1024, Epochs=1, Hidden=256")
    print("="*60)
    with ThreadPool(processes=4) as pool:
        preds_list_B = pool.map(
            partial(fit_predict, y_train=y_train, lr=0.00161, batch_base=1024, epochs=1, hidden_size=256), xs
        )
    score_B_base, score_B_qp = evaluate_model(preds_list_B, valid, y_scaler, y_valid_scaled)

    # ---------------------------------------------------------
    # 최종 리포트 출력
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(" [FINAL A/B TEST REPORT : AS-IS vs TO-BE] ")
    print("="*70)
    print(f" {'Evaluation Metric':<30} | {'AS-IS (Original)':<15} | {'TO-BE (Optimized)':<15}")
    print("-" * 70)
    print(f" 1. Simple 1/N Ensemble         | {score_A_base:.5f}         | {score_B_base:.5f}")
    print(f" 2. Segmented QP (OR)           | {score_A_qp:.5f}         | {score_B_qp:.5f}")
    print("-" * 70)
    print(f" * Improvement (Baseline -> Best) : -{score_A_base - score_B_qp:.5f} (Error Reduction)")
    print("="*70)

if __name__ == '__main__':
    main()