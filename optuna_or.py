import optuna
import subprocess
import re
import time
import logging

import os
os.environ['OMP_NUM_THREADS'] = '1'

# 🚀 [여기에 이 코드를 추가해 주세요]
import sys
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

# 최적화할 대상 스크립트 이름 (수성하신 이름으로 지정)
TARGET_SCRIPT = "or_w_parameter.py"

def run_experiment(lr, batch_base, epoch_multiplier, hidden_size):
    """
    Subprocess를 통해 실제 딥러닝 스크립트를 독립된 환경에서 실행하고,
    출력된 로그에서 최종 Segmented QP RMSLE 점수를 추출해 반환합니다.
    """
    cmd = f"python {TARGET_SCRIPT} --lr {lr} --batch {batch_base} --epochs {epoch_multiplier} --hidden {hidden_size}"
    
    logging.info(f"▶ [실험 시작] 실행 명령어: {cmd}")
    start_time = time.time()
    
    # 🌟 [핵심 수정] encoding='utf-8'과 errors='ignore'를 추가하여 윈도우 이모지 디코딩 에러를 완벽 차단합니다.
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    elapsed_time = (time.time() - start_time) / 60.0
    logging.info(f"▶ [실험 종료] 소요 시간: {elapsed_time:.1f} 분")
    
    # 에러 발생 시 처리
    if result.returncode != 0:
        logging.error("스크립트 실행 중 에러 발생! 해당 파라미터는 실패(Penalty) 처리합니다.")
        logging.error(f"에러 로그 요약:\n{result.stderr[-1500:]}") 
        return 999.0 
        
    # 🌟 이모지가 제거된 영문 텍스트 패턴으로 점수를 파싱합니다.
    match = re.search(r"3\.\s*Segmented QP Optimization.*?:\s*([0-9.]+)", result.stdout)
    if match:
        score = float(match.group(1))
        logging.info(f"🎯 획득한 최적화 Score: {score}")
        return score
    else:
        logging.warning("점수를 파싱하지 못했습니다. 정규표현식 매칭에 실패하여 현재 출력을 로그에 기록합니다.")
        logging.warning(f"실제 스크립트 출력 하이라이트:\n{result.stdout[-1000:]}")
        return 999.0

def objective(trial):
    """
    Optuna가 탐색할 하이퍼파라미터 범위를 설계합니다.
    """
    lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    batch_base = trial.suggest_categorical("batch_base", [1024, 2048, 4096])
    epoch_multiplier = trial.suggest_int("epochs", 1, 3) # 1~3 에포크 탐색
    hidden_size = trial.suggest_categorical("hidden", [128, 192, 256])
    
    score = run_experiment(lr, batch_base, epoch_multiplier, hidden_size)
    return score

def main():
    logging.info("="*60)
    logging.info(" 🌙 밤샘 베이지안 최적화(Bayesian Optimization) 쉘 가동")
    logging.info("="*60)
    
    study = optuna.create_study(
        study_name="mercari_night_run",
        direction="minimize",
        storage="sqlite:///optuna_history.db", 
        load_if_exists=True
    )
    
    # 총 10회 실험 진행 (원하는 대로 조절 가능)
    study.optimize(objective, n_trials=10)
    
    logging.info("="*60)
    logging.info(" 🌅 모든 최적화 실험이 성공적으로 완료되었습니다!")
    logging.info("="*60)
    logging.info(f"🏆 찾은 베스트 최적화 점수: {study.best_value:.5f}")
    logging.info(f"💡 베스트 파라미터 조합: {study.best_params}")

if __name__ == "__main__":
    main()