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
    
    # [LOG ADDED] 스레드별 모델 셋업 로그
    print(f"   [Thread Log] Model Init -> hidden: {hidden_size}, lr: {lr}, Base Batch: {batch_base}")
    
    model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(hidden_size, activation='relu')(model_in)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(learning_rate=lr))
    
    # [LOG ADDED] 에폭별 학습 진행상황 로그
    print(f"   [Thread Log] Start Training (Total Epochs: {epochs})...")
    for i in range(epochs):
        batch_s = batch_base * (2**i)
        print(f"      -> Epoch {i+1}/{epochs} | batch_size={batch_s} running...")
        model.fit(x=X_train, y=y_train, batch_size=batch_s, epochs=1, verbose=0)
        print(f"      -> Epoch {i+1}/{epochs} completed.")
        
    # [LOG ADDED] 추론 시작 로그
    print(f"   [Thread Log] Training finished. Predicting on validation set...")
    return model.predict(X_test)[:, 0]

def optimize_weights_gurobi(preds_matrix, y_true):
    N_models = preds_matrix.shape[1]
    
    # [LOG ADDED] Gurobi 시작 로그
    print(f"   [Gurobi Log] Optimizing weights for {N_models} models on {preds_matrix.shape[0]} samples...")
    
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
        # [LOG ADDED] Gurobi 성공 로그
        print(f"   [Gurobi Log] Optimal weights successfully found!")
        return np.array([w[i].X for i in range(N_models)])
        
    # [LOG ADDED] Gurobi 실패/예외 로그
    print(f"   [Gurobi Log] Warning: Optimization not optimal (Status: {m.status}). Using mean weights.")
    return np.ones(N_models) / N_models

def main():
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("="*60)
    print(" [Phase 1-3] Data Loading and Processing ")
    print("="*60)
    
    # [LOG ADDED]
    print(" [Log] Initializing Feature Union Vectorizer...")
    vectorizer = make_union(
        on_field('name', Tfidf(max_features=100000, token_pattern='\w+')),
        on_field('text', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))),
        on_field(['shipping', 'item_condition_id'], FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=4)
    y_scaler = StandardScaler()

    # [LOG ADDED]
    print(" [Log] Loading input/train.tsv...")
    train_raw = pd.read_table('input/train.tsv')
    
    # [LOG ADDED]
    print(" [Log] Filtering prices > 0 and generating KFold splits...")
    train_raw = train_raw[train_raw['price'] > 0].reset_index(drop=True)
    
    cv = KFold(n_splits=20, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train_raw))
    train = train_raw.iloc[train_ids].copy()
    valid = train_raw.iloc[valid_ids].copy()
    del train_raw
    
    # [LOG ADDED]
    print(" [Log] Scaling target variable (Log1p + StandardScaler)...")
    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    y_valid_scaled = y_scaler.transform(np.log1p(valid['price'].values.reshape(-1, 1)))[:, 0]
    
    print(" [Log] Extracting TF-IDF features for Training set...")
    X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
    del train
    
    # [LOG ADDED]
    print(" [Log] Extracting TF-IDF features for Validation set...")
    X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)

    # [LOG ADDED]
    print(" [Log] Creating Binarized (Boolean) TF-IDF features...")
    Xb_train, Xb_valid = [x.astype(bool).astype(np.float32) for x in [X_train, X_valid]]
    
    # 🌟 불필요한 반복(* 2) 제거 -> (이진 TF-IDF, 일반 TF-IDF) 딱 2세트만 생성
    xs = [[Xb_train, Xb_valid], [X_train, X_valid]] 
    print(" [Log] Data preparation complete. Proceeding to model training.\n")

    # ---------------------------------------------------------
    # 1. 모델 그룹 A (기존 AS-IS 베이스라인 2개) 학습
    # ---------------------------------------------------------
    print("="*60)
    print(" [Step 1] Training Group A (Original AS-IS Models)")
    print("="*60)
    with ThreadPool(processes=2) as pool:
        preds_A = pool.map(partial(fit_predict, y_train=y_train, lr=3e-3, batch_base=2048, epochs=3, hidden_size=192), xs)
    print(" [Log] Group A Training Completed.\n")

    # ---------------------------------------------------------
    # 2. 모델 그룹 B (Optuna가 찾은 이종 다양성 모델 2개) 학습
    # ---------------------------------------------------------
    print("="*60)
    print(" [Step 2] Training Group B (Optuna TO-BE Models for Diversity)")
    print("="*60)
    with ThreadPool(processes=2) as pool:
        # 밤샘 실험에서 찾은 Best 파라미터 적용 완료!
        preds_B = pool.map(partial(fit_predict, y_train=y_train, lr=0.00160955, batch_base=1024, epochs=1, hidden_size=256), xs)
    print(" [Log] Group B Training Completed.\n")

    # ---------------------------------------------------------
    # 3. Gurobi 최적화 (4개 모델 대통합)
    # ---------------------------------------------------------
    print("="*60)
    print(" [Step 3] Gurobi Optimization on ALL 4 Models")
    print("="*60)
    shipping_flags = valid['shipping'].values
    
    # 평가 1: 기존 그룹 A만의 최적화 (대조군 최고 기록)
    print(" [Log] Executing Segmented Gurobi QP for Group A (shipping 0 & 1)...")
    matrix_A = np.column_stack(preds_A)
    w_A0 = optimize_weights_gurobi(matrix_A[shipping_flags == 0], y_valid_scaled[shipping_flags == 0])
    w_A1 = optimize_weights_gurobi(matrix_A[shipping_flags == 1], y_valid_scaled[shipping_flags == 1])
    
    preds_A_seg = np.zeros(len(valid))
    preds_A_seg[shipping_flags == 0] = np.dot(matrix_A[shipping_flags == 0], w_A0)
    preds_A_seg[shipping_flags == 1] = np.dot(matrix_A[shipping_flags == 1], w_A1)
    score_A_best = np.sqrt(mean_squared_log_error(valid['price'], np.expm1(y_scaler.inverse_transform(preds_A_seg.reshape(-1, 1))[:, 0])))

    # 평가 2: 4개 모델 대통합 최적화 (신기록 도전)
    print("\n [Log] Executing Segmented Gurobi QP for ALL Models (shipping 0 & 1)...")
    matrix_ALL = np.column_stack(preds_A + preds_B) # 총 4개 열의 행렬
    w_ALL0 = optimize_weights_gurobi(matrix_ALL[shipping_flags == 0], y_valid_scaled[shipping_flags == 0])
    w_ALL1 = optimize_weights_gurobi(matrix_ALL[shipping_flags == 1], y_valid_scaled[shipping_flags == 1])
    
    preds_ALL_seg = np.zeros(len(valid))
    preds_ALL_seg[shipping_flags == 0] = np.dot(matrix_ALL[shipping_flags == 0], w_ALL0)
    preds_ALL_seg[shipping_flags == 1] = np.dot(matrix_ALL[shipping_flags == 1], w_ALL1)
    score_ALL_best = np.sqrt(mean_squared_log_error(valid['price'], np.expm1(y_scaler.inverse_transform(preds_ALL_seg.reshape(-1, 1))[:, 0])))

    # ---------------------------------------------------------
    # 🌟 [제출용 코드 복사 공간] 황금 가중치 출력 🌟
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" 🌟 [필수 확인] 서버 제출용 황금 가중치 (Golden Weights) 🌟")
    print("="*75)
    print(" 아래의 코드를 복사해서 submit.py의 w_golden 자리에 그대로 붙여넣으세요!\n")
    
    w0_str = ", ".join([f"{w:.6f}" for w in w_ALL0])
    w1_str = ", ".join([f"{w:.6f}" for w in w_ALL1])
    
    print(f"    w_golden_0 = np.array([{w0_str}])")
    print(f"    w_golden_1 = np.array([{w1_str}])")
    print("="*75)

    # ---------------------------------------------------------
    # 최종 리포트 출력
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" 🏆 [GRAND FINALE : OR 최적화 한계 돌파 리포트] 🏆")
    print("="*75)
    print(f" 1. 기존 베이스라인 최고 기록 (A모델 2개 + QP)    : {score_A_best:.5f}")
    print(f" 2. 이종 아키텍처 대통합 기록 (4개 모델 + QP)      : {score_ALL_best:.5f} 🎯")
    print("-" * 75)
    print(f" 💡 성능 한계 돌파 (추가 오차 감소폭)              : -{score_A_best - score_ALL_best:.5f}")
    print("="*75)
    print("\n [Gurobi 가중치 분석 (배송비=0 기준)]")
    print(f" - 강력한 기존 모델 (A) 가중치 합: {np.sum(w_ALL0[:2]):.3f}")
    print(f" - 약하지만 새로운 관점의 모델 (B) 가중치 합: {np.sum(w_ALL0[2:]):.3f}")

if __name__ == '__main__':
    main()