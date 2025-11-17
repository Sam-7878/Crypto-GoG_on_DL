import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
import json
import numpy as np
import analysis.local_metrics.snap_compact as snap  # TNGraphCompat 래퍼(igraph 백엔드 가정)


from pathlib import Path
import sys
# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
# ROOT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.settings import SETTINGS, CHAIN, CHAIN_LABELS


# def build_snap_graph(tx):
#     G = snap.TNGraph.New()
#     node_dict = {}  # Dictionary to map Ethereum addresses to node IDs
#     node_id = 0     # Initial node ID
    
#     for index, row in tx.iterrows():
#         from_addr = row['from']
#         to_addr = row['to']
        
#         if from_addr not in node_dict:
#             node_dict[from_addr] = node_id
#             G.AddNode(node_id)
#             node_id += 1
#         if to_addr not in node_dict:
#             node_dict[to_addr] = node_id
#             G.AddNode(node_id)
#             node_id += 1
        
#         G.AddEdge(node_dict[from_addr], node_dict[to_addr])
    
#     return G

def build_snap_graph(tx):
    """
    빠른 그래프 생성:
      1) 주소를 정수 ID로 factorize
      2) 정점/간선을 일괄 추가
    TNGraphCompat(igraph 백엔드)를 가정, snap.TNGraph.New(N) 지원.
    """
    # 1) 주소 → 정수 ID 매핑 (from/to를 같은 도메인으로 맞춤)
    all_addrs = pd.Index(tx['from']).append(pd.Index(tx['to']))
    uniq = all_addrs.unique()
    # 카테고리 매핑으로 동일한 ID 보장
    cat = pd.Categorical(tx['from'], categories=uniq)
    src = cat.codes.astype(np.int64, copy=False)
    cat = pd.Categorical(tx['to'], categories=uniq)
    dst = cat.codes.astype(np.int64, copy=False)
    n = len(uniq)

    # 2) 정점/간선 일괄 추가
    G = snap.TNGraph.New(n)  # TNGraphCompat.New(N) : N개 정점 미리 추가
    g = G.g if hasattr(G, "g") else G  # igraph.Graph 핸들
    # igraph는 정점 인덱스로 간선 추가 가능
    edges = np.column_stack([src, dst]).tolist()
    if edges:
        g.add_edges(edges)
    return G


# def compute_metrics(G):
#     effective_diameter = snap.GetBfsEffDiam(G, 100, False)
#     clustering_coefficient = snap.GetClustCf(G, -1)
    
#     return effective_diameter, clustering_coefficient

import numpy as np

def _distances_multi_source(g, sources, mode="ALL"):
    # igraph 0.11+: distances(source=..., target=None, mode=...)
    try:
        return np.array(g.distances(source=sources, mode=mode), dtype=float)
    except TypeError:
        pass
    # 일부 구버전: distances(sources=..., targets=None, mode=...)
    try:
        return np.array(g.distances(sources=sources, mode=mode), dtype=float)
    except TypeError:
        pass
    # 최후 수단: 단일 소스로 반복 호출
    rows = []
    for s in sources:
        rows.append(g.shortest_paths(source=s, mode=mode)[0])
    return np.array(rows, dtype=float)


def compute_metrics(G, n_samples=100, undirected=True, rng_seed=0):
    """
    빠른 유효직경 + 클러스터링:
      - 유효직경: 샘플 소스들에 대해 한번에 distances 계산 후 90% 퍼센타일
      - 클러스터링: deg>=2 노드의 로컬 CC 평균
    """
    # --- igraph 핸들 얻기
    g = G.g if hasattr(G, "g") else G
    n = g.vcount()
    if n == 0:
        return 0.0, 0.0

    # --- Effective Diameter
    k = int(min(max(1, n_samples), n))
    rng = np.random.default_rng(rng_seed)
    sources = rng.choice(np.arange(n), size=k, replace=False).tolist()
    mode = "ALL" if undirected else "OUT"
    # 거리 행렬(소스 k × 모든 노드) 한 번에 계산 -> C측 루프
    # dist_mat = np.array(g.distances(sources=sources, mode=mode), dtype=float)
    # python-igraph 0.11.x: keyword는 'source' (리스트 허용)
    dist_mat = _distances_multi_source(g, sources, mode=mode)

    dists = dist_mat[(dist_mat > 0) & np.isfinite(dist_mat)]
    eff_diam = float(np.percentile(dists, 90)) if dists.size else 0.0

    # --- Clustering Coefficient (무방향 기준 권장)
    deg = np.asarray(g.degree(mode="ALL"), dtype=int)
    # mode="zero": deg<2 노드는 0으로 반환 → 평균에서 제외 예정
    local_cc = np.asarray(g.transitivity_local_undirected(mode="zero"), dtype=float)
    mask = deg >= 2
    clust = float(local_cc[mask].mean()) if mask.any() else 0.0

    return eff_diam, clust



def main():
    # chain = 'polygon'
    print("Using chain:", CHAIN)   
    chain = CHAIN
    chain_labels = CHAIN_LABELS
    chain_class = list(chain_labels.Contract.values)

    output_file = './result/'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    stats = []
    for addr in tqdm(chain_class):
        try:
            tx = pd.read_csv(f'./data/transactions/{chain}/{addr}.csv')
            tx['timestamp'] = pd.to_datetime(tx['timestamp'], unit='s')
            end_date = pd.Timestamp('2024-03-01')
            tx = tx[tx['timestamp'] < end_date]

            G = build_snap_graph(tx)
            effective_diameter, clustering_coefficient = compute_metrics(G)
            
            stats.append({
                'Contract': addr,
                'Effective_Diameter': effective_diameter,
                'Clustering_Coefficient': clustering_coefficient
            })
        except Exception as e:
            print(f'Error for address {addr}: {e}')

    df = pd.DataFrame(stats)
    df.to_csv(f'./result/{chain}_advanced_metrics_labels.csv', index=False)

if __name__ == "__main__":
    main()
