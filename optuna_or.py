import optuna
import subprocess
import re
import time
import logging
import sys
import os

os.environ['OMP_NUM_THREADS'] = '1'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 🌟 로깅(Logging) 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("overnight_experiment_log.txt", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

TARGET_SCRIPT = "or_w_parameter.py"

def run_experiment(lr1, batch1, ep1, hid1, lr2, batch2, ep2, hid2):
    # 🌟 8개의 파라미터를 독립적으로 스크립트에 전달합니다.
    cmd = (f"python {TARGET_SCRIPT} "
           f"--lr1 {lr1} --batch1 {batch1} --epochs1 {ep1} --hidden1 {hid1} "
           f"--lr2 {lr2} --batch2 {batch2} --epochs2 {ep2} --hidden2 {hid2}")
    
    logging.info(f"▶ [실험 시작] 실행 명령어: {cmd}")
    start_time = time.time()
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    elapsed_time = (time.time() - start_time) / 60.0
    logging.info(f"▶ [실험 종료] 소요 시간: {elapsed_time:.1f} 분")
    
    if result.returncode != 0:
        logging.error("스크립트 실행 중 에러 발생! 해당 파라미터는 실패(Penalty) 처리합니다.")
        logging.error(f"에러 로그 요약:\n{result.stderr[-1500:]}") 
        return 999.0 
        
    # 기존 코드의 정규표현식 매칭 패턴 (안전장치 추가)
    match = re.search(r"3\.\s*Segmented QP Optimization.*?:\s*([0-9.]+)", result.stdout)
    if not match:
        match = re.search(r"이종 아키텍처 대통합 기록.*?:\s*([0-9.]+)", result.stdout)
        
    if match:
        score = float(match.group(1))
        logging.info(f"🎯 획득한 최적화 Score: {score}")
        return score
    else:
        logging.warning("점수를 파싱하지 못했습니다. 정규표현식 매칭에 실패하여 현재 출력을 로그에 기록합니다.")
        logging.warning(f"실제 스크립트 출력 하이라이트:\n{result.stdout[-1000:]}")
        return 999.0

def objective(trial):
    # 1. 이진 TF-IDF 전용 파라미터 (끝에 1을 붙임)
    lr1 = trial.suggest_float("lr1", 1e-3, 1e-2, log=True)
    batch1 = trial.suggest_categorical("batch1", [1024, 2048, 4096])
    ep1 = trial.suggest_int("epochs1", 1, 3)
    hid1 = trial.suggest_categorical("hidden1", [128, 192, 256])
    
    # 2. 일반 TF-IDF 전용 파라미터 (끝에 2를 붙임)
    lr2 = trial.suggest_float("lr2", 1e-3, 1e-2, log=True)
    batch2 = trial.suggest_categorical("batch2", [1024, 2048, 4096])
    ep2 = trial.suggest_int("epochs2", 1, 3)
    hid2 = trial.suggest_categorical("hidden2", [128, 192, 256])
    
    score = run_experiment(lr1, batch1, ep1, hid1, lr2, batch2, ep2, hid2)
    return score

def main():
    logging.info("="*60)
    logging.info(" 🌙 밤샘 베이지안 최적화(Dual Parameter Tuning) 쉘 가동")
    logging.info("="*60)
    
    study = optuna.create_study(
        study_name="mercari_dual_run",
        direction="minimize",
        storage="sqlite:///optuna_dual_history.db", 
        load_if_exists=True
    )
    
    # 🌟 탐색 공간이 넓어졌으므로 n_trials를 넉넉히(예: 30~50) 주시는 것이 좋습니다.
    study.optimize(objective, n_trials=30)
    
    logging.info("="*60)
    logging.info(" 🌅 모든 최적화 실험이 성공적으로 완료되었습니다!")
    logging.info("="*60)
    logging.info(f"🏆 찾은 베스트 최적화 점수: {study.best_value:.5f}")
    logging.info(f"💡 베스트 파라미터 조합: {study.best_params}")

if __name__ == "__main__":
    main()