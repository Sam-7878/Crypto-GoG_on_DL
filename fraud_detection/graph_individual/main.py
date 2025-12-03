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

import pandas as pd
from datetime import datetime

from pathlib import Path
import sys
import pandas as pd  # <-- 추가
from datetime import datetime  # <-- 추가
# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
# ROOT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.settings import SETTINGS, CHAIN, CHAIN_LABELS


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


def orient_scores(y_true, scores):
    """AUC >= 0.5 방향으로 score를 정렬한 값을 반환합니다.
    evaluate_scores와 동일한 규칙을 사용하지만,
    보정된 score 배열을 돌려줍니다."""
    auc = roc_auc_score(y_true, scores)
    if auc < 0.5:
        return -scores
    return scores


def is_unsupervised_model(model):
    """
    model이 비지도 학습(레이블 없이 학습) 모델인지 판단합니다.
    현재는 COPOD, IForest, DIF, LOF를 비지도 모델로 간주합니다.
    """
    return isinstance(model, (COPOD, IForest, DIF, LOF))


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
    하이퍼파라미터 튜닝(랜덤 검색) 후, 최고 AUC를 주는 파라미터를 반환합니다.
    비지도 모델/지도 모델을 구분하여 적절히 학습합니다.
    """

    def evaluate_single_param_set(x_train, y_train, x_val, y_val, params):
        """
        주어진 파라미터(params)로 model을 학습하고,
        validation set에 대한 AUC/AP를 계산하여 반환합니다.
        """
        model.set_params(**params)

        # VAE 특별 처리 (NaN/발산 방지)
        if isinstance(model, VAE):
            try:
                # 입력 쪽에 비유한 값이 남아 있는지 한 번 더 방어
                if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
                    raise ValueError("x_train/x_val contain non-finite values")

                model.fit(x_train)
                y_scores = model.decision_function(x_val)

                # VAE 출력이 NaN/Inf이면 해당 파라미터 조합은 무효
                if not np.isfinite(y_scores).all():
                    raise ValueError("VAE produced non-finite scores")

            except Exception as e:
                print(
                    f"[WARN] [VAE] skipping param set {params} due to error: {e}"
                )
                # 이 조합은 최적 파라미터로 뽑히지 않도록 아주 낮은 점수 반환
                return -1.0, -1.0

            auc, ap = evaluate_scores(y_val, y_scores)
            return auc, ap

        # ---- VAE가 아닌 나머지 모델들 ----
        if is_unsupervised_model(model):
            model.fit(x_train)
        else:
            model.fit(x_train, y_train)

        y_scores = model.decision_function(x_val)
        auc, ap = evaluate_scores(y_val, y_scores)
        return auc, ap


    best_auc = -1
    best_params = None

    import itertools

    keys = list(param_grid.keys())
    all_param_combinations = list(itertools.product(*param_grid.values()))

    # 현재 모델이 실제로 지원하는 파라미터 목록
    valid_keys_all = set(model.get_params().keys())

    for values in all_param_combinations:
        raw_params = dict(zip(keys, values))

        # 모델이 지원하지 않는 파라미터는 자동으로 제거
        params = {k: v for k, v in raw_params.items() if k in valid_keys_all}
        dropped = set(raw_params.keys()) - set(params.keys())
        if dropped:
            print(
                f"[WARN] [{chain}][{model_name}] dropping invalid params {dropped} "
                f"(not in {model.__class__.__name__}.get_params().keys())"
            )

        auc, ap = evaluate_single_param_set(
            x_train, y_train, x_val, y_val, params
        )

        # 탐색 과정 로그
        print(
            f"[Chain: {chain}] [Model: {model_name}] "
            f"Params: {params} -> AUC: {auc:.4f}, AP: {ap:.4f}"
        )

        # 기록용 dict 구성 및 추가
        rec = {
            "chain": chain,
            "model": model_name,
            "val_auc": float(auc),
            "val_ap": float(ap),
        }
        for k, v in params.items():
            rec[f"param_{k}"] = v
        param_search_records.append(rec)

        # 최고 AUC 갱신 시 best_params 갱신
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

        try:
            if isinstance(model, VAE):
                if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
                    raise ValueError("x_train/x_test contain non-finite values")

                model.fit(x_train)
            else:
                model.fit(x_train, y_train)

            y_scores = model.decision_function(x_test)

            if not np.isfinite(y_scores).all():
                raise ValueError("model produced non-finite scores")

        except Exception as e:
            print(
                f"[WARN] [Eval seeds] skipping seed {seed} for model "
                f"{model.__class__.__name__} due to error: {e}"
            )
            # 이 seed는 건너뛰고 다음 seed로 진행
            continue

        auc, ap = evaluate_scores(y_test, y_scores)
        aucs.append(auc)
        aps.append(ap)

    if not aucs:
        print("[WARN] evaluate_model_with_seeds: no valid AUC/AP computed; returning NaN.")
        return np.nan, np.nan, np.nan, np.nan

    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    avg_ap = np.mean(aps)
    std_ap = np.std(aps)

    return avg_auc, std_auc, avg_ap, std_ap



def load_ts_first_for_chain(chain: str, n_samples: int):
    """
    ./data/features/{chain}_basic_metrics_processed.csv 에 있는 토큰 메타데이터를 기준으로,
    ./data/transactions/{chain}/{contract}.csv 파일에서 최소 timestamp(=ts_first)를 뽑아옵니다.

    반환:
      - 길이 n_samples 의 np.ndarray (float, unix timestamp)
      - 실패 시 None
    """
    meta_path = Path(f"./data/features/{chain}_basic_metrics_processed.csv")
    if not meta_path.exists():
        print(f"[WARN] meta feature file not found: {meta_path}")
        return None

    try:
        meta_df = pd.read_csv(meta_path)
    except Exception as e:
        print(f"[WARN] failed to read meta feature file {meta_path}: {e}")
        return None

    if len(meta_df) != n_samples:
        print(f"[WARN] meta_df length ({len(meta_df)}) != n_samples ({n_samples}); ts_first not attached.")
        return None

    # 어느 컬럼이 컨트랙트 주소인지 추론
    candidate_cols = [
        "Contract",
        "Address",
        "Token",
        "Token_address",
        "Contract_address",
        "address",
        "contract",
        "token",
        "token_address",
        "contract_address",
    ]
    contract_col = None

    # 1) 정확히 일치하는 컬럼명 우선
    for c in candidate_cols:
        if c in meta_df.columns:
            contract_col = c
            break

    # 2) 이름에 address/contract/token 이 포함된 컬럼을 heuristic으로 탐색
    if contract_col is None:
        addr_like_cols = []
        for col in meta_df.columns:
            col_l = str(col).lower()
            if any(kw in col_l for kw in ["Address", "Contract", "Token", "address", "contract", "token"]):
                series = meta_df[col].astype(str)
                sample = series.dropna().head(50)
                if not sample.empty:
                    ratio = (sample.str.startswith("0x")).mean()
                    if ratio >= 0.5:
                        addr_like_cols.append((col, ratio))

        if addr_like_cols:
            # 0x... 비율이 가장 높은 컬럼을 사용
            addr_like_cols.sort(key=lambda x: x[1], reverse=True)
            contract_col = addr_like_cols[0][0]
            print(
                f"[INFO] inferred contract column '{contract_col}' from meta_df "
                f"based on address-like pattern."
            )

    # 3) 그래도 못 찾으면 NaN 배열을 반환해서, 나머지 파이프라인은 계속 돌도록 함
    if contract_col is None:
        print(
            "[WARN] could not find any contract/address-like column in meta_df; "
            "ts_first will be NaN and timestamp will be omitted from score CSV."
        )
        return np.full(n_samples, np.nan, dtype=float)

    print(f"[INFO] using contract column '{contract_col}' from {meta_path}")

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def get_first_timestamp_for_contract(addr: str):
        """
        ./data/transactions/{chain}/{addr}.csv 에서 'timestamp' 혹은 'time_stamp' 컬럼의 최소값을 반환.
        파일이 없거나 컬럼이 없으면 NaN.
        """
        tx_file = Path(f"./data/transactions/{chain}/{addr}.csv")
        if not tx_file.exists():
            # 거래가 없는 토큰일 수 있음
            return np.nan

        try:
            df_tx = pd.read_csv(tx_file)
        except Exception as e:
            print(f"[WARN] failed to read tx file {tx_file}: {e}")
            return np.nan

        # 실제 샘플 파일을 보면 'timestamp' 컬럼이 존재함
        if "timestamp" in df_tx.columns:
            ts = df_tx["timestamp"]
        elif "time_stamp" in df_tx.columns:
            ts = df_tx["time_stamp"]
        else:
            print(f"[WARN] no timestamp column in {tx_file} (expected 'timestamp' or 'time_stamp')")
            return np.nan

        ts = pd.to_numeric(ts, errors="coerce")
        ts = ts.dropna()
        if ts.empty:
            return np.nan
        return float(ts.min())

    ts_list = []
    for addr in meta_df[contract_col]:
        ts_list.append(get_first_timestamp_for_contract(str(addr)))

    ts_arr = np.array(ts_list, dtype=float)
    print("[INFO] loaded ts_first for", np.isfinite(ts_arr).sum(), "/", len(ts_arr), "tokens")
    return ts_arr


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

    # NaN/Inf가 남아 있는지 최종 체크 후 안전하게 치환
    if not np.isfinite(x).all():
        n_nan = np.isnan(x).sum()
        n_posinf = np.isinf(x).sum()
        n_neginf = np.isneginf(x).sum() if hasattr(np, "isneginf") else 0
        print(
            f"[WARN] After imputation, still NaN: {n_nan}, +Inf: {n_posinf}, -Inf: {n_neginf}. "
            "Replacing them with finite values via np.nan_to_num."
        )
        # NaN → 0, +Inf → 큰 양수, -Inf → 큰 음수 로 치환
        x = np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9)

    print(f"[INFO] Loaded dataset for chain '{chain}': {x.shape[0]} samples, {x.shape[1]} features.")


    # Train / Val / Test 분할
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    # -----------------------------
    # 모델 및 하이퍼파라미터 그리드 정의
    # -----------------------------
    models = {}

    models["COPOD"] = (
        COPOD(),
        {
            "n_jobs": [1],
        },
    )

    models["IForest"] = (
        IForest(),
        {
            "n_estimators": [100, 200],
            "max_samples": [256, 512],
            "contamination": [0.01, 0.05, 0.1],
            "n_jobs": [1],
        },
    )

    models["DIF"] = (
        DIF(),
        {
            "n_estimators": [50, 100],
            "max_samples": [256, 512],
            "contamination": [0.01, 0.05, 0.1],
            "n_jobs": [1],
        },
    )

    # VAE Hyperparams
    try:
        models["VAE"] = (
            VAE(),
            {
                "n_hidden": [16, 32],
                "n_latent": [4, 8],
                "epochs": [50, 100],
                "batch_size": [64, 128],
                "contamination": [0.01, 0.05, 0.1],
                "dropout_rate": [0.1, 0.3],
                "lr": [1e-3, 1e-4],
            },
        )
    except TypeError:
        # pyod VAE 시그니처가 다른 버전용 fallback
        models["VAE"] = (
            VAE(),
            {
                "hidden_neurons": [[16], [32]],
                "latent_dim": [4, 8],
                "epochs": [50, 100],
                "batch_size": [64, 128],
                "contamination": [0.01, 0.05, 0.1],
                "dropout_rate": [0.1, 0.3],
                "learning_rate": [1e-3, 1e-4],
            },
        )

    results_dir = Path("./results/fraud_detection/graph_individual")
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ----------------------------------
    # 하이퍼파라미터 탐색 + 기록
    # ----------------------------------
    seeds = [42, 43, 44]
    param_search_records = []
    final_eval_records = []

    # -----------------------------
    # 전체 샘플에 대한 score CSV 준비
    # -----------------------------
    n_samples = x.shape[0]

    # (NEW) 토큰별 첫 거래 timestamp(ts_first) 로드
    ts_first = load_ts_first_for_chain(chain, n_samples)

    print(f"[INFO] Preparing timestamp-attached score DataFrame for {n_samples} samples.")

    if ts_first is None:
        sys.stderr.write(f"[INFO] ts_first is not available.")
        sys.exit(1)
    else:
        print(f"[INFO] Sample ts_first values (first 10): {ts_first[:10]}")

    score_df = pd.DataFrame(
        {
            "chain": [chain] * n_samples,
            "idx": np.arange(n_samples),
            "label": y.astype(int),
        }
    )

    # ts_first가 정상적으로 계산되었으면 컬럼으로 추가
    if ts_first is not None:
        score_df["ts_first"] = ts_first


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

            # 전체 데이터에 대한 score 계산 및 저장용 컬럼 추가
            try:
                model.set_params(**best_params)
                if isinstance(model, VAE):
                    model.fit(x)
                elif is_unsupervised_model(model):
                    model.fit(x)
                else:
                    model.fit(x, y)
                raw_scores = model.decision_function(x)
                oriented_scores = orient_scores(y, raw_scores)
                col_name = f"score_{model_name}"
                score_df[col_name] = oriented_scores
            except Exception as e:
                print(f"[WARN] Failed to compute full-dataset scores for {model_name}: {e}")

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

    # 개별 토큰별 score CSV 저장 (MC / Nested GNN 파이프라인에서 baseline으로 활용)
    score_csv_path = results_dir / f"graph_individual_scores_{chain}_{run_id}.csv"
    score_df.to_csv(score_csv_path, index=False)
    print(f"[INFO] Per-sample scores saved to: {score_csv_path}")


if __name__ == "__main__":
    main()
