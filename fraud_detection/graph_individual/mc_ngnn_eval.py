#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chain-wise anomaly detection evaluation with time-based split and bootstrap CIs.

Input: per-chain token-level CSVs with at least:
  - 'chain'            : chain name (ethereum|bsc|polygon ...)
  - 'token'            : token contract (id/hash)
  - 'ts_first'         : first-activity timestamp (ISO or epoch)  -> used for time split
  - 'label'            : 1=fraud/abnormal, 0=normal  (if missing, runs unsupervised reporting)
Optional score/feature columns:
  - Baseline GoG features: columns starting with 'dw_' (DeepWalk), or simple stats (deg_*, dens, reciprocity, clustering)
  - MC signals: 'local_zmax', 'fisher_p_global', 'rwr_score', 'temporal_post'  (names align with 앞서 제안한 MC 파이프라인)
  - Nested GNN score: 'ngnn_score'  (precomputed score or logit); or 'ngnn_emb_*' for linear head fallback.
  - PyOD graph_individual scores: 'score_*' columns from fraud_detection/graph_individual/main.py

Usage:
  python experiment_chain_eval.py \
    --csv data/per_chain/ethereum_tokens.csv \
    --time_col ts_first \
    --time_split 0.8 \
    --bootstrap 1000 \
    --method baseline gog_mc nested_gnn pyod \
    --out results/eth_eval.csv
"""

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from pathlib import Path
import sys
import pandas as pd  # <-- 추가
from datetime import datetime  # <-- 추가
# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent
# ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.settings import SETTINGS, CHAIN, CHAIN_LABELS

# -------------------------
# Utilities
# -------------------------
def parse_time(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)) and x > 10_000_000:  # epoch seconds-ish
        try:
            return pd.to_datetime(int(x), unit='s', utc=True)
        except Exception:
            return pd.to_datetime(x, utc=True, errors='coerce')
    return pd.to_datetime(x, utc=True, errors='coerce')


def ensure_binary_label(y: pd.Series):
    if y.isna().any():
        # treat NaN as unlabeled -> drop for supervised metrics
        m = ~y.isna()
        return y[m].astype(int), m
    return y.astype(int), y.notna()


def zscore(df: pd.DataFrame, cols: List[str]):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols].values)
    return X, scaler


def safe_metric_auc(y_true, y_score):
    # Handle edge cases (all same label -> AUC undefined)
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def safe_metric_ap(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


def stratified_bootstrap_idx(y: np.ndarray, n_boot: int):
    """
    Y가 이진 레이블일 때, class별로 bootstrap sampling을 수행하는 index를 yield.
    """
    y = np.asarray(y)
    n = len(y)
    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}

    rng = np.random.default_rng(1729)
    for _ in range(n_boot):
        idx_list = []
        for c in classes:
            idx_c = idx_by_class[c]
            if len(idx_c) == 0:
                continue
            draw = rng.choice(idx_c, size=len(idx_c), replace=True)
            idx_list.append(draw)
        if not idx_list:
            yield np.arange(n)
        else:
            yield np.concatenate(idx_list)


def ci_mean(arr: np.ndarray, alpha=0.05):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    lo = np.quantile(arr, alpha/2)
    hi = np.quantile(arr, 1 - alpha/2)
    return float(np.mean(arr)), float(lo), float(hi)

# -------------------------
# Method: Baseline GoG
# -------------------------
def run_baseline_gog(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """
    If DeepWalk-like columns 'dw_*' exist, train a simple logistic head.
    Else fall back to light stats: ['deg_in','deg_out','dens','reciprocity','clustering'] if present.
    If no features found, return normalized degree as heuristic.
    """
    feat_cols = [c for c in train.columns if c.startswith("dw_")]
    if not feat_cols:
        candidates = ['deg_in','deg_out','deg_total','dens','reciprocity','clustering']
        feat_cols = [c for c in candidates if c in train.columns]

    if not feat_cols:
        # last resort: uniform score
        sys.stderr.write("[baseline] No feature columns found; returning zeros.\n")
        return np.zeros(len(test), dtype=float)

    Xtr, scaler = zscore(train, feat_cols)
    Xte = scaler.transform(test[feat_cols].values)

    # Supervised head if labels exist
    if 'label' in train.columns:
        ytr, mtr = ensure_binary_label(train['label'])
        Xtr = Xtr[mtr.values]
        if len(np.unique(ytr)) >= 2:
            clf = LogisticRegression(max_iter=200, solver='lbfgs', class_weight='balanced')
            clf.fit(Xtr, ytr)
            return clf.predict_proba(Xte)[:, 1]

    # Otherwise unsupervised: simple norm of embedding
    scr = np.linalg.norm(Xte, axis=1)
    return (scr - scr.min()) / (scr.ptp() + 1e-9)

# -------------------------
# Method: MC pipeline
# -------------------------
def run_mc_pipeline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """
    Monte Carlo / RWR 기반 신호(local_zmax, fisher_p_global, rwr_score, temporal_post)를 합치는 간단한 베이스라인.
    - train에서 z-score 정규화를 학습하고
    - label이 있으면 supervised head (LogReg), 없으면 weighted sum + min-max.
    """
    exist = set(train.columns) | set(test.columns)
    cols = []
    if 'local_zmax' in exist: cols.append('local_zmax')
    if 'fisher_p_global' in exist: cols.append('fisher_p_global')
    if 'rwr_score' in exist: cols.append('rwr_score')
    if 'temporal_post' in exist: cols.append('temporal_post')

    if not cols:
        sys.stderr.write("[mc] No MC columns found; returning zeros.\n")
        return np.zeros(len(test), dtype=float)

    # transform fisher p to -log10
    def transform(df):
        X = df[cols].copy()
        if 'fisher_p_global' in X.columns:
            X['neglogp'] = -np.log10(np.clip(X['fisher_p_global'].values, 1e-300, 1.0))
            X.drop(columns=['fisher_p_global'], inplace=True)
        return X

    Xtr_raw = transform(train)
    Xte_raw = transform(test)

    feat_cols = list(Xtr_raw.columns)
    Xtr, scaler = zscore(pd.concat([train[[]], Xtr_raw], axis=1), feat_cols)  # reuse zscore helper
    Xte = scaler.transform(Xte_raw.values)

    # Supervised if labels available (and at least one pos/neg)
    if 'label' in train.columns:
        ytr, mtr = ensure_binary_label(train['label'])
        Xtr = Xtr[mtr.values]
        if len(np.unique(ytr)) >= 2:
            clf = LogisticRegression(max_iter=200, solver='lbfgs', class_weight='balanced')
            clf.fit(Xtr, ytr)
            return clf.predict_proba(Xte)[:, 1]

    # Unsupervised fallback: isotonic calibration of simple weighted sum
    w = np.ones(Xtr.shape[1], dtype=float) / Xtr.shape[1]
    s_tr = (Xtr * w).sum(axis=1)
    s_te = (Xte * w).sum(axis=1)
    return (s_te - s_te.min()) / (s_te.ptp() + 1e-9)

# -------------------------
# Method: Nested GNN
# -------------------------
def run_nested_gnn(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """
    If a precomputed 'ngnn_score' column exists -> use directly.
    Else if 'ngnn_emb_*' exists -> train a simple logistic head.
    Else -> fallback: combine local/global structural features as proxy.
    """
    if 'ngnn_score' in test.columns:
        s = test['ngnn_score'].values.astype(float)
        # normalize to [0,1] just in case
        s = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-9)
        return s

    emb_cols = [c for c in train.columns if c.startswith('ngnn_emb_')]
    if emb_cols:
        Xtr, scaler = zscore(train, emb_cols)
        Xte = scaler.transform(test[emb_cols].values)
        if 'label' in train.columns:
            ytr, mtr = ensure_binary_label(train['label'])
            Xtr = Xtr[mtr.values]
            if len(np.unique(ytr)) >= 2:
                clf = LogisticRegression(max_iter=200, solver='lbfgs', class_weight='balanced')
                clf.fit(Xtr, ytr)
                return clf.predict_proba(Xte)[:, 1]
        # unsupervised fallback: l2 norm
        scr = np.linalg.norm(Xte, axis=1)
        return (scr - scr.min()) / (scr.ptp() + 1e-9)

    # Proxy fallback: use graph stats if available
    candidates = ['deg_in','deg_out','deg_total','dens','reciprocity','clustering']
    feat_cols = [c for c in candidates if c in train.columns]
    if not feat_cols:
        sys.stderr.write("[ngnn] No NGNN columns found; returning zeros.\n")
        return np.zeros(len(test), dtype=float)
    Xtr, scaler = zscore(train, feat_cols)
    Xte = scaler.transform(test[feat_cols].values)
    if 'label' in train.columns:
        ytr, mtr = ensure_binary_label(train['label'])
        Xtr = Xtr[mtr.values]
        if len(np.unique(ytr)) >= 2:
            clf = LogisticRegression(max_iter=200, solver='lbfgs', class_weight='balanced')
            clf.fit(Xtr, ytr)
            return clf.predict_proba(Xte)[:, 1]
    scr = np.linalg.norm(Xte, axis=1)
    return (scr - scr.min()) / (scr.ptp() + 1e-9)

# -------------------------
# Method: PyOD Graph-Individual baseline
# -------------------------
def run_pyod_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Use precomputed per-token PyOD scores from graph_individual/main.py.

    Expects one or more columns whose name starts with 'score_' (e.g., 'score_COPOD', 'score_IForest', ...).
    If labels are available, fits a logistic regression head on train scores.
    Otherwise, returns normalized mean of available score columns.
    """
    # find candidate score columns
    cand_cols = [c for c in test.columns if c.startswith("score_")]
    if not cand_cols:
        sys.stderr.write("[pyod] No 'score_*' columns found; returning zeros.\n")
        return np.zeros(len(test), dtype=float)

    # align columns in train/test
    feat_cols = [c for c in cand_cols if c in train.columns]
    if not feat_cols:
        sys.stderr.write("[pyod] No overlapping 'score_*' columns in train; returning zeros.\n")
        return np.zeros(len(test), dtype=float)

    Xtr = train[feat_cols].values.astype(float)
    Xte = test[feat_cols].values.astype(float)

    if 'label' in train.columns:
        ytr, mtr = ensure_binary_label(train['label'])
        Xtr = Xtr[mtr.values]
        if len(np.unique(ytr)) >= 2:
            clf = LogisticRegression(max_iter=200, solver='lbfgs', class_weight='balanced')
            clf.fit(Xtr, ytr)
            return clf.predict_proba(Xte)[:, 1]

    # unsupervised fallback: normalized mean of score columns
    s = Xte.mean(axis=1)
    return (s - s.min()) / (s.ptp() + 1e-9)


# -------------------------
# Bootstrap evaluation
# -------------------------
def evaluate(y_true: np.ndarray, scores: Dict[str, np.ndarray], n_boot: int) -> pd.DataFrame:
    """
    Returns a tall DataFrame with mean and 95% CI for AUC and AP per method.
    """
    res = []
    for name, s in scores.items():
        aucs, aps = [], []
        # base (full test) metrics
        auc_full = safe_metric_auc(y_true, s)
        ap_full  = safe_metric_ap(y_true, s)

        for idx in stratified_bootstrap_idx(y_true, n_boot):
            yb = y_true[idx]
            sb = s[idx]
            aucs.append(safe_metric_auc(yb, sb))
            aps.append(safe_metric_ap(yb, sb))

        auc_mean, auc_lo, auc_hi = ci_mean(np.array(aucs))
        ap_mean, ap_lo, ap_hi    = ci_mean(np.array(aps))

        res.append(
            {
                "method": name,
                "AUC_full": auc_full,
                "AUC_mean": auc_mean,
                "AUC_lo": auc_lo,
                "AUC_hi": auc_hi,
                "AP_full": ap_full,
                "AP_mean": ap_mean,
                "AP_lo": ap_lo,
                "AP_hi": ap_hi,
            }
        )

    return pd.DataFrame(res)

# -------------------------
# Time-based split
# -------------------------
def time_split_df(df: pd.DataFrame, time_col: str, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    frac 비율을 기준으로 time_col의 quantile을 계산하고, 그 이전을 train, 이후를 test로 나눕니다.
    """
    if time_col not in df.columns:
        raise ValueError(f"time_col {time_col} not in DataFrame")

    t = df[time_col].apply(parse_time)
    df = df.copy()
    df[time_col + "_parsed"] = t

    # drop rows with invalid time
    m = ~df[time_col + "_parsed"].isna()
    df = df[m]
    t = df[time_col + "_parsed"]

    cut = t.quantile(frac)
    train = df[t <= cut].copy()
    test  = df[t > cut].copy()
    return train, test, cut

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Per-chain token table CSV")
    ap.add_argument("--time_col", default="ts_first", help="Time column for split")
    ap.add_argument("--time_split", type=float, default=0.8, help="Train fraction by time (0<frac<1)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="# of bootstrap iterations")
    ap.add_argument("--method", nargs="+", default=["baseline","gog_mc","nested_gnn","pyod"],
                    choices=["baseline","gog_mc","nested_gnn","pyod"])
    ap.add_argument("--out", default=None, help="Output CSV for results")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.time_col not in df.columns:
        sys.stderr.write(f"[error] time column '{args.time_col}' not in CSV.\n")
        sys.exit(1)

    if 'label' not in df.columns:
        sys.stderr.write("[warn] No label column. Supervised metrics will drop NaNs and may be undefined.\n")

    # Time split
    train, test, cut = time_split_df(df, args.time_col, args.time_split)
    print(f"[info] time split @ {args.time_col} quantile={args.time_split:.2f}, cut={cut}")
    print(f"[info] train={len(train)}, test={len(test)}")

    # Build scores per method
    scores = {}
    if "baseline" in args.method:
        scores["baseline_gog"] = run_baseline_gog(train, test)
    if "gog_mc" in args.method:
        scores["mc"] = run_mc_pipeline(train, test)
    if "pyod" in args.method:
        scores["pyod_graph_individual"] = run_pyod_baseline(train, test)
    if "nested_gnn" in args.method:
        scores["nested_gnn"] = run_nested_gnn(train, test)

    # Labels
    if 'label' not in test.columns:
        print("[warn] No labels in test set; cannot compute AUC/AP.")
        # Dump placeholder
        out_df = pd.DataFrame([{"method": k, "AUC_full": np.nan, "AP_full": np.nan} for k in scores.keys()])
    else:
        y_true, mask = ensure_binary_label(test['label'])
        # drop unlabeled rows if any
        valid = mask.values
        for k in scores:
            scores[k] = np.asarray(scores[k])[valid]
        y_true = y_true.values

        out_df = evaluate(y_true, scores, n_boot=args.bootstrap)

    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"[ok] saved: {args.out}")
    else:
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
