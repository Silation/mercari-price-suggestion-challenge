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

import argparse
import gurobipy as gp
from gurobipy import GRB

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

def fit_predict(xs, y_train, lr, batch_base, epochs, hidden_size) -> np.ndarray:
    X_train, X_test = xs
    
    model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(hidden_size, activation='relu')(model_in) 
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(learning_rate=lr)) 
    
    print(f"   ▶ [모델 학습 시작] 데이터 크기: {X_train.shape} ... (총 {epochs} Epoch 진행)")
    for i in range(epochs): 
        batch_s = batch_base * (2**i) 
        model.fit(x=X_train, y=y_train, batch_size=batch_s, epochs=1, verbose=0)
    return model.predict(X_test)[:, 0]

def optimize_weights_gurobi(preds_matrix, y_true):
    N_models = preds_matrix.shape[1]
    
    print("   ▶ [Gurobi] 수학적 모델 생성 중...")
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0) 
    env.start()
    m = gp.Model("Ensemble_Weight_Optimization", env=env)
    
    w = m.addVars(N_models, lb=0.0, ub=1.0, name="w")
    m.addConstr(gp.quicksum(w[i] for i in range(N_models)) == 1.0, "SumToOne")
    
    print("   ▶ [Gurobi] 7.4만개 행렬 연산 및 이차 계획법(QP) 셋업 중...")
    PTP = np.dot(preds_matrix.T, preds_matrix) 
    PTy = np.dot(preds_matrix.T, y_true)       
    
    obj = gp.QuadExpr()
    for i in range(N_models):
        for j in range(N_models):
            obj += PTP[i, j] * w[i] * w[j]
    for i in range(N_models):
        obj -= 2.0 * PTy[i] * w[i]
        
    m.setObjective(obj, GRB.MINIMIZE)
    
    print("   ▶ [Gurobi] QP 솔버 구동 시작!")
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        opt_weights = [w[i].X for i in range(N_models)]
        print(f"   ▶ [Gurobi] 최적해 도출 성공! 가중치 비율: {np.round(opt_weights, 3)}")
        return np.array(opt_weights)
    else:
        print("   ▶ [Gurobi] 최적해 실패. 단순 평균으로 대체합니다.")
        return np.ones(N_models) / N_models

def main(args):
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
        train_raw = pd.read_table('input/train.tsv')
        train_raw = train_raw[train_raw['price'] > 0].reset_index(drop=True)
        
        cv = KFold(n_splits=20, shuffle=True, random_state=42)
        train_ids, valid_ids = next(cv.split(train_raw))
        train = train_raw.iloc[train_ids].copy()
        valid = train_raw.iloc[valid_ids].copy()
        del train_raw 
        
        y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
        y_valid_scaled = y_scaler.transform(np.log1p(valid['price'].values.reshape(-1, 1)))[:, 0]
        
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        del train 
        
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)

    print()
    print("="*60)
    print(" [Phase 3] Valid 데이터 전처리 진행")
    print("="*60)
    with timer('Phase 3 전체'):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)

    print()
    print("="*60)
    print(" [Phase 4] 멀티프로세싱 모델 독립 학습 (Dual Parameters)")
    print("="*60)
    
    Xb_train, Xb_valid = [x.astype(np.bool_).astype(np.float32) for x in [X_train, X_valid]]
    xs = [[Xb_train, Xb_valid], [X_train, X_valid]]
    
    # 순차적으로 실행하여 멀티프로세싱 충돌 방지 및 파라미터 분리 적용
    preds_list = []
    
    # 1. 이진 TF-IDF 전용 파라미터 학습
    print("   ▶ [모델 1] 이진 TF-IDF 학습 시작...")
    pred1 = fit_predict(xs[0], y_train, lr=args.lr1, batch_base=args.batch1, epochs=args.epochs1, hidden_size=args.hidden1)
    preds_list.append(pred1)
    
    # 2. 일반 TF-IDF 전용 파라미터 학습
    print("   ▶ [모델 2] 일반 TF-IDF 학습 시작...")
    pred2 = fit_predict(xs[1], y_train, lr=args.lr2, batch_base=args.batch2, epochs=args.epochs2, hidden_size=args.hidden2)
    preds_list.append(pred2)
    
    print("\n" + "="*60)
    print(" [Phase 5] OR Strategy: Segmented QP Optimization")
    print("="*60)
    
    preds_matrix = np.column_stack(preds_list)
    
    preds_mean = np.mean(preds_list, axis=0)
    preds_mean_inv = np.expm1(y_scaler.inverse_transform(preds_mean.reshape(-1, 1))[:, 0])
    score_baseline = np.sqrt(mean_squared_log_error(valid['price'], preds_mean_inv))
    
    shipping_flags = valid['shipping'].values
    
    print("\n  ▶ [Segmented QP] Running Gurobi solvers for each shipping condition...")
    weights_ship0 = optimize_weights_gurobi(
        preds_matrix[shipping_flags == 0], y_valid_scaled[shipping_flags == 0]
    )
    weights_ship1 = optimize_weights_gurobi(
        preds_matrix[shipping_flags == 1], y_valid_scaled[shipping_flags == 1]
    )
    
    preds_segmented = np.zeros(len(valid))
    preds_segmented[shipping_flags == 0] = np.dot(preds_matrix[shipping_flags == 0], weights_ship0)
    preds_segmented[shipping_flags == 1] = np.dot(preds_matrix[shipping_flags == 1], weights_ship1)
    
    preds_seg_inv = np.expm1(y_scaler.inverse_transform(preds_segmented.reshape(-1, 1))[:, 0])
    score_segmented_qp = np.sqrt(mean_squared_log_error(valid['price'], preds_seg_inv))
    
    print("\n" + "="*60)
    print(" [FINAL EXPERIMENT REPORT] ")
    print("="*60)
    print(f" 1. Simple Ensemble Score (1/N)       : {score_baseline:.5f}")
    print(f" 3. Segmented QP Optimization         : {score_segmented_qp:.5f}")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 이진 TF-IDF용 (1)
    parser.add_argument('--lr1', type=float, default=0.003)
    parser.add_argument('--batch1', type=int, default=2048)
    parser.add_argument('--epochs1', type=int, default=3)
    parser.add_argument('--hidden1', type=int, default=192)
    # 일반 TF-IDF용 (2)
    parser.add_argument('--lr2', type=float, default=0.003)
    parser.add_argument('--batch2', type=int, default=2048)
    parser.add_argument('--epochs2', type=int, default=3)
    parser.add_argument('--hidden2', type=int, default=192)
    
    args = parser.parse_args()
    main(args)