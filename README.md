# Mercari Price Suggestion Challenge
> **Machine Learning & Operations Research Fusion Project**

본 프로젝트는 딥러닝 기반의 가격 예측 알고리즘(Keras/TensorFlow)과 수리 최적화 기법(SciPy SLSQP / Gurobi QP)을 융합하여 단일 머신러닝 모델의 성능 한계를 돌파하는 MLOps 및 OR 융합 프레임워크를 구축한 연구입니다.

* ⚠️ input 폴더 압축 해제 이후 진행 필수
* input/train.tsv가 존재해야 진행 가능

---

## 📂 핵심 파일 및 실행 가이드 (Core Scripts)

프로젝트의 각 단계별 핵심 소스 코드 및 검증 스크립트는 다음과 같이 구성되어 있습니다. 목적에 따라 해당 스크립트를 실행하시기 바랍니다.

### 1. 기존 베이스라인 모델 검증 (`original.py`)
* **설명:** Kaggle 대회 당시 최고의 성적을 거두었던 집단지성 기반의 엑스퍼트 튜닝(Expert-Tuned) 신경망 아키텍처 모델입니다.
* **실행 목적:** 최적화 적용 전 본 프로젝트의 대조군(AS-IS) 성능 평점을 단독으로 확인하고자 할 때 사용합니다.
* **실행 방법:**
  ```bash
  python original.py
  ```

### 2. 제안 최적화 파이프라인 및 진행 상황 (`grand_ensemble.py`)
* **설명:** 본 팀의 핵심 기여도가 반영된 최신 진행 상황이 담긴 소스 코드입니다. 베이지안 최적화(Optuna)를 통해 도출된 이종 아키텍처(Heterogeneous Pool) 모델들을 생성하고, 이들의 예측값을 조건부 이차 계획법(Segmented QP)으로 결합하여 한계를 돌파(TO-BE)하는 실험 파이프라인입니다.
* **실행 목적:** 1/N 단순 앙상블 대비 수리 최적화 계층이 적용되었을 때의 확정적인 오차 감소폭(`-0.00214`) 및 신기록 달성 여부를 검증할 때 사용합니다.
* **실행 방법:**
  ```bash
  python grand_ensemble.py
  ```

### 3. 캐글 최종 서버 제출 모듈 (`submit.py`)
* **설명:** 모의고사(Validation) 단계를 넘어 실제 Kaggle 리더보드에 점수를 기록하기 위해 최적화된 실전 제출용 스크립트입니다.
* **실행 목적:** 정답이 비어있는 진짜 평가용 데이터셋(`test_stg2.tsv`)을 로드하여 8개 이종 모델의 예측값을 황금 가중치 비율로 결합하고, 규격에 맞는 최종 `submission.csv` 파일을 생성하고자 할 때 실행합니다.
* **실행 방법:**
  ```bash
  python submit.py
  ```

### 4. 문제 이해 및 보조 분석 자료 (`Gemini-interpretation/`)
* **설명:** 20만 차원의 거대 희소 행렬 데이터 구조 분석, 인코딩 가이드라인, 하이퍼파라미터 탐색 로그, 윈도우/캐글 구형 환경 호환성 트러블슈팅 내역 등 프로젝트 전반의 기술적 인사이트가 정리된 디렉토리입니다.
* **실행 목적:** 수리적 모델링의 배경 논리나 시스템 오류 해결 과정을 참고하고자 할 때 탐색합니다.

---

## 🛠️ 환경 세팅 및 필수 라이브러리 (Requirements)
본 코드는 로컬 환경 및 캐글 커널(Kaggle Notebook) 구형/신형 환경에 모두 대응되도록 방탄 처리(Robust Exception Handling)되어 있습니다.

* **Language:** Python 3.6+
* **Frameworks:** TensorFlow 1.x/2.x, Keras 2.1.3+
* **Optimization Solver:** SciPy (SLSQP Algorithm) 및 Gurobi (gurobipy)
* **Data Science Tools:** Scikit-learn, Pandas, NumPy

---

## 💡 주요 기여도 (Project Contributions)
1. **White-box 최적화 융합:** 딥러닝의 비과학적 1/N 평균 결합 관행을 깨고, 배송비 조건에 따라 분할 최적화하는 **Segmented QP** 계층을 도입했습니다.
2. **이종 앙상블 다양성 확보:** 단독 성능이 떨어지는 하이퍼파라미터 모델을 폐기하지 않고, 수리 최적화의 훌륭한 재료(Solution Space)로 승화시켜 **시스템 한계선을 완벽하게 돌파(0.387 → 0.384)**해 냈습니다.