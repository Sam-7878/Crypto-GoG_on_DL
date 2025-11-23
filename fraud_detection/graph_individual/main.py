import numpy as np
import torch
import random
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.lof import LOF  # 현재는 사용하지 않지만 남겨둡니다.
from pyod.models.vae import VAE
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from fraud_detection.graph_individual.utils import GraphDatasetGenerator

from pathlib import Path
import sys

# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.settings import SETTINGS, CHAIN, CHAIN_LABELS

import pandas as pd
from datetime import datetime


def evaluate_scores(y_true, scores):
    """이상치 score 방향을 자동 보정해서 AUC/AP를 계산"""
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    # 만약 AUC가 0.5보다 낮으면, 점수 방향이 반대일 가능성이 크므로 뒤집어서 다시 계산
    if auc < 0.5:
        scores = -scores
        auc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)

    return auc, ap


def is_unsupervised_deep_model(model):
    # 필요하면 AutoEncoder, ALAD 같은 다른 딥러닝 비지도 모델도 여기에 추가 가능
    return isinstance(model, VAE)


def eval_roc_auc(label, score):
    roc_auc = roc_auc_score(y_true=label, y_score=score)
    if roc_auc < 0.5:
        score = [1 - s for s in score]
        roc_auc = roc_auc_score(y_true=label, y_score=score)
    return roc_auc


def eval_average_precision(label, score):
    return average_precision_score(y_true=label, y_score=score)


def tune_and_find_best_params(
    model,
    param_grid,
    x_train,
    y_train,
    x_val,
    y_val,
    model_name: str,
    chain: str,
    param_search_records: list,
):
    """
    - 주어진 param_grid에 대해 validation AUC가 가장 높은 파라미터를 찾고
    - 각 파라미터 조합에 대한 결과를 param_search_records 리스트에 기록합니다.
    """
    best_auc = -np.inf
    best_params = None

    for params in param_grid:   # 이미 list[dict] 이므로 그대로 순회
        model.set_params(**params)

        if isinstance(model, VAE):
            # VAE는 비지도 → y 사용 X
            model.fit(x_train)
        else:
            model.fit(x_train, y_train)

        y_val_scores = model.decision_function(x_val)
        auc, ap = evaluate_scores(y_val, y_val_scores)

        # CSV 저장용 record 추가
        rec = {
            "chain": chain,
            "model": model_name,
            "auc": float(auc),
            "ap": float(ap),
        }
        # 파라미터들도 같이 기록 (모델마다 파라미터 구성이 달라서 컬럼은 union이 됩니다)
        for k, v in params.items():
            rec[k] = v
        param_search_records.append(rec)

        if auc > best_auc:
            best_auc = auc
            best_params = params

    return best_params


# DIF처럼 “score 방향이 뒤집혀 있는 모델”도 자동으로 AUC ≥ 0.5 쪽으로 맞춰서 평가합니다.
# VAE도 만약 score 방향이 거꾸로라면 약간 개선된 값이 나올 수 있습니다.
def evaluate_model_with_seeds(model, best_params, x, y, seeds):
    aucs, aps = [], []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=seed, stratify=y
        )

        model.set_params(**best_params)

        if isinstance(model, VAE):
            model.fit(x_train)
        else:
            model.fit(x_train, y_train)

        y_scores = model.decision_function(x_test)
        auc, ap = evaluate_scores(y_test, y_scores)

        aucs.append(auc)
        aps.append(ap)

    return np.mean(aucs), np.std(aucs), np.mean(aps), np.std(aps)


def main():
    print("Using chain:", CHAIN)
    chain = CHAIN

    # -----------------------------
    # 데이터 로드
    # -----------------------------
    dataset_generator = GraphDatasetGenerator(
        f'./data/features/{chain}_basic_metrics_processed.csv'
    )
    data_list = dataset_generator.get_pyg_data_list()
    x = torch.cat([data.x for data in data_list], dim=0).numpy()
    y = torch.cat(
        [data.y.unsqueeze(0) if data.y.dim() == 0 else data.y for data in data_list]
    ).numpy()

    # NaN 처리
    imputer = SimpleImputer(strategy='mean')
    x = imputer.fit_transform(x)

    # Train / Val / Test 분할
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.1, random_state=42
    )

    num_features = x.shape[1]
    hidden_size = min(20, num_features // 2)

    try:
        models = {
            "COPOD": (
                COPOD(),
                [{"contamination": f} for f in np.linspace(0.01, 0.1, 10)],
            ),

            "Isolation Forest": (
                IForest(),
                [
                    {"n_estimators": n, "max_samples": s}
                    for n in [100, 200]
                    for s in [256, 512]
                ],
            ),

            "DIF": (
                DIF(),
                [{"contamination": f} for f in np.linspace(0.01, 0.05, 5)],
            ),

            "VAE": (
                VAE(
                    encoder_neuron_list=[hidden_size],
                    decoder_neuron_list=[hidden_size],
                    contamination=0.1,
                ),
                [
                    {
                        "encoder_neuron_list": [n],
                        "decoder_neuron_list": [n],
                        "contamination": f,
                    }
                    for n in [max(1, hidden_size // 2), hidden_size, hidden_size * 2]
                    for f in np.linspace(0.1, 0.3, 3)
                ],
            ),
        }

    except TypeError as e:
        raise RuntimeError(
            f"VAE 초기화 실패: 현재 설치된 pyod 버전의 VAE 인자 이름을 확인하세요. 원본 에러: {e}"
        )

    # ----------------------------------
    # 결과 저장 폴더 및 타임스탬프 준비
    # ----------------------------------
    results_dir = Path("./results/fraud_detection/graph_individual")
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ----------------------------------
    # 하이퍼파라미터 탐색 + 기록
    # ----------------------------------
    seeds = [42, 43, 44]
    param_search_records = []
    final_eval_records = []

    for model_name, (model, param_grid) in models.items():
        best_params = tune_and_find_best_params(
            model,
            param_grid,
            x_train,
            y_train,
            x_val,
            y_val,
            model_name=model_name,
            chain=chain,
            param_search_records=param_search_records,
        )

        if best_params:
            avg_auc, std_auc, avg_ap, std_ap = evaluate_model_with_seeds(
                model, best_params, x, y, seeds
            )

            print(
                f"{model_name} Results: "
                f"Average AUC = {avg_auc:.4f} ± {std_auc:.4f}, "
                f"Average AP = {avg_ap:.4f} ± {std_ap:.4f}"
            )

            # 최종 평가 결과 기록
            rec = {
                "chain": chain,
                "model": model_name,
                "final_avg_auc": float(avg_auc),
                "final_std_auc": float(std_auc),
                "final_avg_ap": float(avg_ap),
                "final_std_ap": float(std_ap),
                "seeds": ",".join(map(str, seeds)),
            }
            # best_params도 함께 저장
            for k, v in best_params.items():
                rec[f"best_{k}"] = v
            final_eval_records.append(rec)

        else:
            print(f"{model_name} failed to find suitable parameters.")

    # ----------------------------------
    # CSV 파일로 저장
    # ----------------------------------
    param_csv_path = results_dir / f"graph_individual_param_search_{chain}_{run_id}.csv"
    final_csv_path = results_dir / f"graph_individual_final_eval_{chain}_{run_id}.csv"

    if param_search_records:
        pd.DataFrame(param_search_records).to_csv(param_csv_path, index=False)
        print(f"[INFO] Hyperparameter search results saved to: {param_csv_path}")

    if final_eval_records:
        pd.DataFrame(final_eval_records).to_csv(final_csv_path, index=False)
        print(f"[INFO] Final evaluation results saved to: {final_csv_path}")


if __name__ == "__main__":
    main()
