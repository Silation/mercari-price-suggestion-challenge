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

# 🌟 [새로 추가된 OR 라이브러리]
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
    out = ks.layers.Dense(hidden_size, activation='relu')(model_in) # 🌟 은닉층 크기 변수화
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(learning_rate=lr)) # 🌟 학습률 변수화
    
    print(f"   ▶ [모델 학습 시작] 데이터 크기: {X_train.shape} ... (총 {epochs} Epoch 진행)")
    for i in range(epochs): # 🌟 에포크 수 변수화
        # 초기 배치 사이즈(batch_base)부터 시작해서 에포크마다 2배씩 증가
        batch_s = batch_base * (2**i) 
        model.fit(x=X_train, y=y_train, batch_size=batch_s, epochs=1, verbose=0)
    return model.predict(X_test)[:, 0]

# 🌟 [OR 최적화 함수 설계] Gurobi를 이용한 앙상블 가중치 도출
def optimize_weights_gurobi(preds_matrix, y_true):
    N_models = preds_matrix.shape[1]
    
    print("   ▶ [Gurobi] 수학적 모델 생성 중...")
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0) # Gurobi 내부 로그 출력 끄기
    env.start()
    m = gp.Model("Ensemble_Weight_Optimization", env=env)
    
    # 의사결정 변수: 4개 모델의 가중치 w (0 ~ 1 사이 실수)
    w = m.addVars(N_models, lb=0.0, ub=1.0, name="w")
    
    # 제약조건: 가중치의 총합은 1이어야 함
    m.addConstr(gp.quicksum(w[i] for i in range(N_models)) == 1.0, "SumToOne")
    
    # 목적 함수: (y - P*w)^2 최소화
    # 연산 속도 극대화를 위해 선형대수 행렬(P^T * P) 전개 방식을 사용합니다.
    print("   ▶ [Gurobi] 7.4만개 행렬 연산 및 이차 계획법(QP) 셋업 중...")
    PTP = np.dot(preds_matrix.T, preds_matrix) # shape: (4, 4)
    PTy = np.dot(preds_matrix.T, y_true)       # shape: (4,)
    
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

def main():
    # 🌟 argparse를 통해 쉘 스크립트가 넘겨주는 값을 받음
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=192) # 2번 질문(네트워크 튜닝)을 위한 추가 인자
    args = parser.parse_args()
    
    
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
        print(f"원본 데이터 로드 완료: 총 {len(train_raw)}개 행")

        cv = KFold(n_splits=20, shuffle=True, random_state=42)
        train_ids, valid_ids = next(cv.split(train_raw))
        train = train_raw.iloc[train_ids].copy()
        valid = train_raw.iloc[valid_ids].copy()
        print(f"데이터 분할 완료 -> 학습용(Train): {len(train)}개 / 검증용(Valid): {len(valid)}개")
        del train_raw # 메모리 확보
        
        # 2. 타겟(Price) 스케일링
        y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
        # Gurobi 최적화를 위한 Valid 정답 스케일링
        y_valid_scaled = y_scaler.transform(np.log1p(valid['price'].values.reshape(-1, 1)))[:, 0]
        
        # 3. 텍스트 TF-IDF 벡터화
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        del train # 벡터화가 끝났으므로 텍스트 원본 삭제
        
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)

    print()
    print("="*60)
    print(" [Phase 3] Valid 데이터 전처리 진행")
    print("="*60)
    with timer('Phase 3 전체'):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
        print(f"Valid 피처 변환 완료: 차원 크기 = {X_valid.shape}")

    print()
    print("="*60)
    print(" [Phase 4] 멀티프로세싱 4개 모델 병렬 학습")
    print("="*60)
    with ThreadPool(processes=4) as pool:
        Xb_train, Xb_valid = [x.astype(np.bool_).astype(np.float32) for x in [X_train, X_valid]]
        xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
        
        # preds_list = pool.map(partial(fit_predict, y_train=y_train), xs)
        
        # 🌟 인자들을 partial을 통해 fit_predict로 쏴줌
        preds_list = pool.map(
            partial(fit_predict, y_train=y_train, lr=args.lr, batch_base=args.batch, 
                    epochs=args.epochs, hidden_size=args.hidden), 
            xs
        )
    
    # 1. 기존 방식 (단순 평균: 1/N)
    print("\n" + "="*60)
    print(" [Phase 5] OR Strategy: Segmented QP Optimization")
    print("="*60)
    
    preds_matrix = np.column_stack(preds_list)
    
    # 1. 기존 방식 (단순 평균: 1/N)
    preds_mean = np.mean(preds_list, axis=0)
    preds_mean_inv = np.expm1(y_scaler.inverse_transform(preds_mean.reshape(-1, 1))[:, 0])
    score_baseline = np.sqrt(mean_squared_log_error(valid['price'], preds_mean_inv))
    
    # 2. Global QP
    opt_weights_global = optimize_weights_gurobi(preds_matrix, y_valid_scaled)
    preds_global = np.dot(preds_matrix, opt_weights_global)
    preds_global_inv = np.expm1(y_scaler.inverse_transform(preds_global.reshape(-1, 1))[:, 0])
    score_global_qp = np.sqrt(mean_squared_log_error(valid['price'], preds_global_inv))
    
    # 3. Segmented QP
    shipping_flags = valid['shipping'].values
    
    print("\n   ▶ [Segmented QP] Running Gurobi solvers for each shipping condition...")
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
    
    # 🌟 [이모지 완전 제거 및 텍스트 표준화]
    print("\n" + "="*60)
    print(" [FINAL EXPERIMENT REPORT] ")
    print("="*60)
    print(f" 1. Simple Ensemble Score (1/N)       : {score_baseline:.5f}")
    print(f" 2. Global QP Optimization            : {score_global_qp:.5f}")
    print(f" 3. Segmented QP Optimization         : {score_segmented_qp:.5f}")
    print(f"\n Improvement (1/N vs Segmented QP)  : -{score_baseline - score_segmented_qp:.5f}")
    print("="*60)


# 🌟 [OR 최적화 확장] 혼합 정수 이차 계획법(MIQP) 기반 앙상블 가지치기
def prune_and_optimize_weights_gurobi(preds_matrix, y_true, k_select=2):
    """
    전체 N개의 모델 중, 정확히 k_select 개의 모델만 선택하여 가중치를 부여하는 최적화 모델
    """
    N_models = preds_matrix.shape[1]
    
    print(f"   ▶ [Gurobi MIQP] {N_models}개 모델 중 최적의 {k_select}개 가지치기(Pruning) 시작...")
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Ensemble_Pruning", env=env)
    
    # 의사결정 변수 1: 연속형 가중치 w (0 ~ 1)
    w = m.addVars(N_models, lb=0.0, ub=1.0, name="w", vtype=GRB.CONTINUOUS)
    
    # 의사결정 변수 2: 이진 변수 z (해당 모델을 선택했으면 1, 버렸으면 0)
    z = m.addVars(N_models, name="z", vtype=GRB.BINARY)
    
    # 제약조건 1: 정확히 k_select 개의 모델만 선택해야 함
    m.addConstr(gp.quicksum(z[i] for i in range(N_models)) == k_select, "Select_K_Models")
    
    # 제약조건 2: 가중치의 총합은 1이어야 함
    m.addConstr(gp.quicksum(w[i] for i in range(N_models)) == 1.0, "SumToOne")
    
    # 제약조건 3: 논리적 연결 제약 (Big-M 제약)
    # 모델을 선택하지 않았다면(z=0), 그 모델의 가중치(w)는 반드시 0이어야 함
    # 모델을 선택했다면(z=1), 가중치(w)는 최대 1까지 가질 수 있음
    for i in range(N_models):
        m.addConstr(w[i] <= z[i], f"Link_w_z_{i}")
        
    # 목적 함수: (y - P*w)^2 최소화 (기존 QP와 동일한 선형대수 전개식)
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
        opt_weights = [w[i].X for i in range(N_models)]
        selected_models = [i for i in range(N_models) if z[i].X > 0.5]
        print(f"   ▶ [Gurobi MIQP] 가지치기 성공! 선택된 모델 인덱스: {selected_models}")
        print(f"   ▶ [Gurobi MIQP] 가지치기 가중치 비율: {np.round(opt_weights, 3)}")
        return np.array(opt_weights)
    else:
        print("   ▶ [Gurobi MIQP] 최적해 실패. 단순 평균으로 대체합니다.")
        return np.ones(N_models) / N_models

if __name__ == '__main__':
    main()