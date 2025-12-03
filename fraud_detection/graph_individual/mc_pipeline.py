# mc_pipeline.py
import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
import networkx as nx

from fraud_detection.graph_individual.mc_pipeline_skeleton import (
    estimate_local_metrics,
    surprise_pvalues,
    hypergeo_pval,
    edge_weight_from_p,
    mc_rwr,
    smc_update,
    fuse_scores,
)

# -----------------------------
# TokenGraph 래퍼 (skeleton에서 기대하는 인터페이스)
# -----------------------------
class TokenGraph:
    def __init__(self, edges):
        # edges: list of (src, dst)
        self.G = nx.DiGraph()
        self.G.add_edges_from(edges)

    def num_edges(self):
        return self.G.number_of_edges()

    def num_nodes(self):
        return self.G.number_of_nodes()

    def sample_node_by_degree(self):
        nodes, degs = zip(*self.G.degree()) if self.G.number_of_nodes() > 0 else ([], [])
        if not nodes:
            raise RuntimeError("TokenGraph is empty")
        return random.choices(nodes, weights=degs, k=1)[0]

    def sample_two_neighbors(self, v):
        nbrs = list(self.G.successors(v))
        if len(nbrs) < 2:
            # fallback: 자기 자신 두 번
            return (v, v)
        return tuple(random.sample(nbrs, 2))

    def has_edge(self, u, w):
        return self.G.has_edge(u, w)

    def sample_edge(self):
        edges = list(self.G.edges())
        if not edges:
            raise RuntimeError("TokenGraph has no edges")
        return random.choice(edges)

# -----------------------------
# 1) YAML 로딩
# -----------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

# -----------------------------
# 2) transaction → 토큰별 TokenGraph 만들기
# -----------------------------
def build_token_graphs_from_tx(tx_df, token_col, src_col, dst_col):
    """
    tx_df: transaction DataFrame
    token_col: 토큰 주소/ID 컬럼 이름
    src_col, dst_col: from / to address
    """
    edges_by_token = defaultdict(list)
    for _, row in tx_df.iterrows():
        token = row[token_col]
        src = row[src_col]
        dst = row[dst_col]
        if pd.isna(token) or pd.isna(src) or pd.isna(dst):
            continue
        edges_by_token[token].append((src, dst))

    token_graphs = {}
    for token, edges in edges_by_token.items():
        token_graphs[token] = TokenGraph(edges)
    return token_graphs

# -----------------------------
# 3) 로컬 MC metrics → local_zmax
# -----------------------------
def compute_local_mc_features(token_graphs, cfg_local):
    """
    token_graphs: dict[token] -> TokenGraph
    """
    records = []
    for token, tg in token_graphs.items():
        try:
            m = estimate_local_metrics(tg, cfg_local)
        except Exception as e:
            print(f"[WARN] local MC failed for token={token}: {e}")
            m = {"clustering_hat": np.nan, "reciprocity_hat": np.nan}
        records.append(
            {
                "token": token,
                "clustering_hat": m.get("clustering_hat", np.nan),
                "reciprocity_hat": m.get("reciprocity_hat", np.nan),
            }
        )
    df = pd.DataFrame(records)

    # z-score
    for col in ["clustering_hat", "reciprocity_hat"]:
        mu = df[col].mean()
        sigma = df[col].std() or 1.0
        df[col + "_z"] = (df[col] - mu) / sigma

    # local_zmax = max |z|
    df["local_zmax"] = df[["clustering_hat_z", "reciprocity_hat_z"]].abs().max(axis=1)
    return df

# -----------------------------
# 4) 토큰-토큰 글로벌 그래프 + MC-RWR
# -----------------------------
def compute_global_rwr_scores(token_address_sets, labels, cfg_rwr):
    """
    token_address_sets: dict[token] -> set(address)
    labels: dict[token] -> 0/1
    """
    tokens = list(token_address_sets.keys())
    # 전체 고유 주소 수
    all_addrs = set()
    for s in token_address_sets.values():
        all_addrs |= s
    nU = len(all_addrs)

    # 토큰-토큰 weighted graph
    G = nx.Graph()
    G.add_nodes_from(tokens)

    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            A = tokens[i]
            B = tokens[j]
            setA = token_address_sets[A]
            setB = token_address_sets[B]
            if not setA or not setB:
                continue
            inter = setA & setB
            k = len(inter)
            if k == 0:
                continue
            nA = len(setA)
            nB = len(setB)
            p = hypergeo_pval(nA, nB, nU, k)
            w = edge_weight_from_p(p)
            if w > 0:
                G.add_edge(A, B, weight=w)

    # seeds: label==1인 토큰
    seeds = [t for t, y in labels.items() if y == 1]
    if not seeds:
        print("[WARN] no positive labels; global RWR will be all zeros.")
        return {t: 0.0 for t in tokens}

    # mc_rwr는 directed/weighted graph를 가정하므로 래핑
    class RWGraph:
        def __init__(self, G):
            self.G = G

        def neighbors(self, v):
            return list(self.G.neighbors(v))

        def weight(self, v, u):
            return self.G[v][u].get("weight", 1.0)

    rwG = RWGraph(G)
    rwr = mc_rwr(
        rwG,
        seeds,
        alpha=cfg_rwr.get("alpha", 0.2),
        length=cfg_rwr.get("length", 12),
        walks_per_seed=cfg_rwr.get("walks_per_seed", 2000),
        gamma=cfg_rwr.get("gamma", 0.5),
    )

    # 없는 토큰은 0
    rwr_scores = {t: rwr.get(t, 0.0) for t in tokens}
    return rwr_scores

# -----------------------------
# 5) 시간축 SMC → temporal_post
# -----------------------------
def compute_temporal_posterior(df_local, rwr_scores, df_meta, cfg_temporal):
    """
    df_local: token, local_zmax 포함
    rwr_scores: dict[token] -> rwr_score
    df_meta: token + ts_first 포함 DataFrame
    """
    df = df_meta.merge(df_local[["token", "local_zmax"]], on="token", how="left")
    df["rwr_score"] = df["token"].map(rwr_scores).fillna(0.0)

    # 시간순으로 정렬
    df = df.sort_values("ts_first").reset_index(drop=True)

    # 초기 particles
    n_particles = cfg_temporal.get("n_particles", 200)
    particles = [{"z": 0, "w": 1.0 / n_particles} for _ in range(n_particles)]
    # 절반은 z=1로 시작하게 해도 됨
    for i in range(n_particles // 2):
        particles[i]["z"] = 1

    temporal_post = []
    for _, row in df.iterrows():
        obs = [row["local_zmax"], row["rwr_score"]]
        particles, ess = smc_update(particles, obs, cfg_temporal)
        # posterior P(z=1)
        p_abn = sum(p["w"] for p in particles if p["z"] == 1)
        temporal_post.append(p_abn)

    df["temporal_post"] = temporal_post
    return df[["token", "local_zmax", "rwr_score", "temporal_post"]]

# -----------------------------
# 6) 전체 파이프라인
# -----------------------------
def run_mc_pipeline(cfg_path):
    cfg = load_config(cfg_path)
    chain = cfg.get("chain", "bsc")

    # 1) 메타/label 로딩
    meta_path = cfg["basic_metrics_csv"]
    df_meta = pd.read_csv(meta_path)

    # token 식별 컬럼명은 YAML에 맞춰 주세요
    token_col = cfg.get("token_col", "token_address")
    label_col = cfg.get("label_col", "label")

    # 2) tx 로딩 후 TokenGraph 만들기
    tx_path = cfg["tx_csv"]
    tx_df = pd.read_csv(tx_path)

    src_col = cfg.get("src_col", "from_address")
    dst_col = cfg.get("dst_col", "to_address")

    token_graphs = build_token_graphs_from_tx(tx_df, token_col, src_col, dst_col)

    # 3) 로컬 MC
    df_local = compute_local_mc_features(token_graphs, cfg["local"])

    # 4) 토큰-토큰 글로벌 그래프용 address set & label
    token_address_sets = defaultdict(set)
    for _, row in tx_df[[token_col, src_col, dst_col]].iterrows():
        token = row[token_col]
        if pd.isna(token):
            continue
        token_address_sets[token].add(row[src_col])
        token_address_sets[token].add(row[dst_col])

    labels = dict(zip(df_meta[token_col], df_meta[label_col]))
    rwr_scores = compute_global_rwr_scores(token_address_sets, labels, cfg["rwr"])

    # 5) temporal posterior
    # df_meta 에 ts_first 컬럼이 있다고 가정
    df_meta_time = df_meta[[token_col, "ts_first"]].rename(columns={token_col: "token"})
    df_temporal = compute_temporal_posterior(df_local.rename(columns={token_col: "token"}),
                                             rwr_scores,
                                             df_meta_time,
                                             cfg["temporal"])

    # 6) 결과 merge & 저장
    df_mc = df_temporal.copy()
    df_mc["chain"] = chain

    out_path = cfg["mc_features_csv"]
    df_mc.to_csv(out_path, index=False)
    print(f"[ok] MC features saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="mc_pipeline.yaml path")
    args = parser.parse_args()
    run_mc_pipeline(args.config)
