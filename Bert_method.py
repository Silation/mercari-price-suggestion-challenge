import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
conda_prefix = os.environ.get('CONDA_PREFIX', '/home/tlfgja8/miniconda3/envs/ds-bert')
os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={conda_prefix}"

import numpy as np
import pandas as pd
import tensorflow as tf
from operator import itemgetter
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import KFold  # 🌟 이 부분이 꼭 있어야 함!
from transformers import AutoTokenizer, TFAutoModel

# ---------------------------------------------------------
# 1. 전처리 유틸리티
# ---------------------------------------------------------
def preprocess_df(df):
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    return df[['name', 'text', 'shipping', 'item_condition_id']]

def preprocess_text_only(df):
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    return (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))

def to_records(df): return df.to_dict(orient='records')
def on_field(f, *vec): return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

# ---------------------------------------------------------
# 2. 제너레이터 (OOM 에러 방지)
# ---------------------------------------------------------
class SparseDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_sparse, y=None, batch_size=2048):
        self.X_sparse = X_sparse.tocsr()
        self.y = y
        self.batch_size = batch_size
        self.num_samples = X_sparse.shape[0]

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_x = self.X_sparse[start_idx:end_idx].toarray()
        if self.y is not None:
            return batch_x, self.y[start_idx:end_idx]
        return batch_x

# ---------------------------------------------------------
# 3. BERT 오프라인 추출 및 MLP 모델 정의
# ---------------------------------------------------------
def extract_and_save_bert_features(texts, tokenizer, filename, batch_size=512):
    if os.path.exists(filename):
        print(f"   [Info] '{filename}' 파일이 존재하여 로드합니다.")
        return np.load(filename)
        
    print(f"   [Extract] '{filename}' 임베딩 추출 시작 (총 {len(texts)}건)...")
    bert_base = TFAutoModel.from_pretrained('distilbert-base-uncased')
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='tf')
        cls_output = bert_base(**inputs)[0][:, 0, :] 
        embeddings.append(cls_output.numpy())
        if (i // batch_size) % 50 == 0: 
            print(f"      ... {i}/{len(texts)} 건 완료")
            
    final_embeddings = np.vstack(embeddings)
    np.save(filename, final_embeddings)
    del bert_base
    tf.keras.backend.clear_session() 
    return final_embeddings

def build_simple_mlp(input_dim, lr, hidden_size):
    model_in = tf.keras.Input(shape=(input_dim,), dtype='float32')
    out = tf.keras.layers.Dense(hidden_size, activation='relu')(model_in)
    out = tf.keras.layers.Dropout(0.1)(out) 
    out = tf.keras.layers.Dense(64, activation='relu')(out)
    out = tf.keras.layers.Dense(64, activation='relu')(out)
    out = tf.keras.layers.Dense(1)(out)
    
    model = tf.keras.Model(inputs=model_in, outputs=out)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    return model

def fit_predict(xs, y_train, lr, batch_base, epochs, hidden_size, loss_fn='mean_squared_error') -> np.ndarray:
    X_train, X_test = xs
    
    # [LOG ADDED] 스레드별 모델 셋업 로그 (손실 함수 추가)
    print(f"   [Thread Log] Model Init -> hidden: {hidden_size}, lr: {lr:.6f}, Base Batch: {batch_base}, Loss: {loss_fn}")
    
    model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(hidden_size, activation='relu')(model_in)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    
    # 🌟 손실 함수(loss_fn)를 동적으로 받아서 컴파일!
    model.compile(loss=loss_fn, optimizer=ks.optimizers.Adam(learning_rate=lr))
    
    print(f"   [Thread Log] Start Training (Total Epochs: {epochs})...")
    for i in range(epochs):
        batch_s = batch_base * (2**i)
        print(f"      -> Epoch {i+1}/{epochs} | batch_size={batch_s} running...")
        model.fit(x=X_train, y=y_train, batch_size=batch_s, epochs=1, verbose=0)
        print(f"      -> Epoch {i+1}/{epochs} completed.")
        
    print(f"   [Thread Log] Training finished. Predicting on validation set...")
    return model.predict(X_test)[:, 0]

# ---------------------------------------------------------
# 4. 메인 파이프라인
# ---------------------------------------------------------
def main():
    print("="*70)
    print(" 🚀 [최종 제출] 하이브리드 모델 4개 순차 학습 & 평가 파이프라인 가동")
    print("="*70)
    
    print(" [1] 훈련(140만) 및 테스트(350만) 데이터 로드 중...")
    train_raw = pd.read_table('input/train.tsv')
    train_raw = train_raw[train_raw['price'] > 0].reset_index(drop=True)
    
    # 🌟 [복구] 기존에 몇 시간 걸려 뽑아둔 1,407,577개짜리 train_bert.npy를 
    # 그대로 재활용하기 위해, 당시와 완벽히 똑같은 K-Fold 분할을 적용합니다!
    cv = KFold(n_splits=20, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train_raw))
    train = train_raw.iloc[train_ids].copy() # 정확히 1,407,577개로 맞춰짐!
    del train_raw # 메모리 확보
    
    test = pd.read_table('input/test_stg2.tsv') 
    
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    
    print(" [2] TF-IDF 추출 중...")
    vectorizer = make_union(
        on_field('name', Tfidf(max_features=50000, token_pattern=r'\w+')),
        on_field('text', Tfidf(max_features=50000, token_pattern=r'\w+', ngram_range=(1, 2))),
        on_field(['shipping', 'item_condition_id'], FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=4)
        
    X_train_tfidf_ori = vectorizer.fit_transform(preprocess_df(train)).astype(np.float32)
    X_test_tfidf_ori = vectorizer.transform(preprocess_df(test)).astype(np.float32)

    # 🌟 이진화(Binarized) TF-IDF 생성 
    X_train_tfidf_bin = X_train_tfidf_ori.astype(bool).astype(np.float32)
    X_test_tfidf_bin = X_test_tfidf_ori.astype(bool).astype(np.float32)

    print(" [3] BERT 임베딩 로드/추출 중...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 🌟 기존 파일(train_bert.npy)이 존재하므로 1초 만에 로드하고 넘어갑니다!
    X_train_bert = extract_and_save_bert_features(preprocess_text_only(train), tokenizer, 'train_bert.npy')
    
    # 단, 350만 개 테스트 데이터(test_stg2)는 처음 보는 데이터이므로 여기서 새롭게 추출해야 합니다. 
    # (이 부분만 4080 기준으로 약 15~20분 정도 소요됩니다.)
    X_test_bert = extract_and_save_bert_features(preprocess_text_only(test), tokenizer, 'test_bert.npy')

    print(" [4] TF-IDF와 BERT 결합 중 (CSR Matrix)...")
    # Original + BERT
    X_train_combined_ori = hstack([X_train_tfidf_ori, csr_matrix(X_train_bert)]).tocsr()
    X_test_combined_ori = hstack([X_test_tfidf_ori, csr_matrix(X_test_bert)]).tocsr()
    
    # Binarized + BERT
    X_train_combined_bin = hstack([X_train_tfidf_bin, csr_matrix(X_train_bert)]).tocsr()
    X_test_combined_bin = hstack([X_test_tfidf_bin, csr_matrix(X_test_bert)]).tocsr()

    print("\n [5] 4개 모델 순차 학습 및 테스트셋 예측 시작...")
    
    tasks = [
        # (훈련 데이터, 테스트 데이터, 타겟, lr, batch, epochs, hidden)
        (X_train_combined_bin, X_test_combined_bin, y_train, 3e-3, 2048, 3, 192), # Group A 1
        (X_train_combined_ori, X_test_combined_ori, y_train, 3e-3, 2048, 3, 192), # Group A 2
        (X_train_combined_bin, X_test_combined_bin, y_train, 0.001524, 1024, 3, 256), # Group B 1
        (X_train_combined_ori, X_test_combined_ori, y_train, 0.002626, 2048, 2, 128)  # Group B 2
    ]

    all_test_preds = []
    
    for idx, (x_tr, x_te, y_tr, lr, batch_base, epochs, hidden_size) in enumerate(tasks):
        print(f"\n" + "="*50)
        print(f" 🚀 [Model {idx+1}/4] 학습 시작 (hidden={hidden_size}, lr={lr})")
        
        model = build_simple_mlp(input_dim=x_tr.shape[1], lr=lr, hidden_size=hidden_size)
        
        # 모델 학습
        for i in range(epochs):
            batch_s = batch_base * (2**i)
            print(f"    -> Epoch {i+1}/{epochs} (Batch: {batch_s})")
            train_gen = SparseDataGenerator(x_tr, y_tr, batch_size=batch_s)
            model.fit(train_gen, epochs=1, verbose=1, workers=4, use_multiprocessing=True)
            
        # 🌟 모델 학습이 끝나자마자 바로 350만 개 테스트셋 예측
        print(f"    -> [Model {idx+1}/4] 350만 개 테스트 데이터 예측 중...")
        test_gen = SparseDataGenerator(x_te, y=None, batch_size=4096)
        pred = model.predict(test_gen)[:, 0]
        all_test_preds.append(pred)
        
        # OOM 방지를 위해 GPU 메모리 강제 반환
        del model
        tf.keras.backend.clear_session()
        print("="*50)

    print("\n [6] 4개 모델 훈련/예측 완료! 황금 가중치(Golden Weights) 적용 중...")
    
    # 4개의 예측 결과를 (3460725, 4) 형태의 행렬로 합침
    matrix_all = np.column_stack(all_test_preds)
    shipping_flags = test['shipping'].values
    
    # 미리 구했던 황금 가중치
    w_golden_0 = np.array([0.241525, 0.321700, 0.172541, 0.264233]) 
    w_golden_1 = np.array([0.288720, 0.377159, 0.162786, 0.171336]) 
    
    # 가중치 곱 연산
    preds_final = np.zeros(len(test))
    preds_final[shipping_flags == 0] = np.dot(matrix_all[shipping_flags == 0], w_golden_0)
    preds_final[shipping_flags == 1] = np.dot(matrix_all[shipping_flags == 1], w_golden_1)
    
    # 로그 스케일로 예측된 값을 원래 달러($) 가격으로 복원
    final_price = np.expm1(y_scaler.inverse_transform(preds_final.reshape(-1, 1))[:, 0])

    print("\n [7] submission.csv 생성 완료!")
    submission = pd.DataFrame({'test_id': test['test_id'], 'price': final_price})
    submission.to_csv('submission.csv', index=False)
    
    print("="*70)
    print(" ✅ 준비 완료! 만들어진 submission.csv를 캐글 Dataset으로 올리고 제출하세요!")
    print("="*70)

if __name__ == '__main__':
    main()