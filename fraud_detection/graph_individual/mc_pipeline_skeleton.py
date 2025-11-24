# pipeline_skeleton.py
import hashlib, math, random, time
from collections import Counter, defaultdict

# --- 유틸 ---
def ci_half_width(p, n, z=1.96):
    var = max(p*(1-p), 1e-9)
    return z*math.sqrt(var/n)

# --- 로컬 MC (개요) ---
def estimate_local_metrics(token_graph, cfg):
    E = token_graph.num_edges()
    V = token_graph.num_nodes()
    # 샘플 수 설정
    wedge_samples = min(cfg["wedge_samples_max"],
                        max(cfg["wedge_samples_min"], cfg["wedge_samples_per_edge"]*E))
    # 예: 클러스터링 근사 (wedge 샘플링)
    tri_hits = 0
    for _ in range(wedge_samples):
        # 임의의 중심 v와 두 이웃 u,w를 샘플 → u-w 연결 여부 체크
        v = token_graph.sample_node_by_degree()
        u, w = token_graph.sample_two_neighbors(v)
        if token_graph.has_edge(u, w): tri_hits += 1
    clustering_hat = tri_hits / max(wedge_samples, 1)
    # reciprocity 근사
    R = cfg["reciprocity_edge_samples"]
    recip_hits = 0
    for _ in range(R):
        (a,b) = token_graph.sample_edge()
        if token_graph.has_edge(b,a): recip_hits += 1
    reciprocity_hat = recip_hits / max(R, 1)
    return {
        "clustering_hat": clustering_hat,
        "reciprocity_hat": reciprocity_hat,
        # 필요 시 모티프 추가 …
    }

# --- 널 모델 놀람도 (Chung-Lu) ---
def surprise_pvalues(observed_metrics, null_sampler, replicas=50):
    sims = null_sampler(run_replicas=replicas)  # 각 replica에서 동일 지표 산출
    pvals = {}
    for k, obs in observed_metrics.items():
        dist = [sim[k] for sim in sims if k in sim]
        rank = sum(1 for x in dist if x >= obs)
        p = (rank+1) / (len(dist)+1)
        pvals[k] = p
    return pvals

# --- 글로벌: 후보 생성(MinHash/LSH 대체 자리) + 정확 p-value ---
def hypergeo_pval(nA, nB, nU, k):
    # nU: 전체 고유주소, nA: 토큰A 고유주소, nB: 토큰B 고유주소, k: 교집합
    # 큰 규모에서는 정확 계산 대신 근사/MC로 대체
    # 여기선 자리표시용 간략 근사
    from math import comb
    num = comb(nA, k)*comb(nU-nA, nB-k)
    den = comb(nU, nB)
    return max(min(num/den, 1.0), 0.0)

def edge_weight_from_p(p, clip=(0.0,10.0)):
    if p <= 0: return clip[1]
    w = -math.log10(p)
    return min(max(w, clip[0]), clip[1])

# --- MC-RWR ---
def mc_rwr(G, seeds, alpha=0.2, length=12, walks_per_seed=2000, gamma=0.5):
    hits = Counter()
    seed_list = list(seeds)
    for s in seed_list:
        for _ in range(walks_per_seed):
            v = s
            for _ in range(length):
                if random.random() < alpha:
                    v = random.choice(seed_list)
                else:
                    nbrs = G.neighbors(v)
                    if not nbrs: break
                    # 가중치^gamma 비례 선택
                    weights = [max(G.weight(v,u),1e-9)**gamma for u in nbrs]
                    v = random.choices(nbrs, weights=weights, k=1)[0]
                hits[v] += 1
    total = sum(hits.values()) or 1
    return {v: hits[v]/total for v in hits}

# --- SMC (개요) ---
def smc_update(particles, obs, cfg):
    # particles: [{'z':0/1, 'w':float}, ...]
    # obs: 표준화된 관측 벡터 -> 상태별 우도 계산 (Student-t 근사)
    def like(z):
        # z==1일 때 mean shift / var scale
        shift = cfg["abnormal_mean_shift"] if z==1 else 0.0
        scale = cfg["abnormal_var_scale"] if z==1 else 1.0
        # 간단히 정규 근사
        import numpy as np, math
        x = np.array(obs)
        mu = shift
        var = scale
        # 독립 가정한 대략적 점수 (자리표시)
        return math.exp(-0.5*((x-mu)**2/var).sum())
    # 가중치 업데이트
    for p in particles:
        p["w"] *= like(p["z"])
    # 정규화
    W = sum(p["w"] for p in particles) or 1.0
    for p in particles: p["w"] /= W
    # ESS 체크
    ess = 1.0 / sum((p["w"]**2 for p in particles))
    return particles, ess

# --- 융합 ---
def fuse_scores(local_zmax, fisher_p, rwr, rwr_ci_low, temporal_post, weights):
    # 비감독: 가중합
    score = (weights["local"] * (local_zmax/3.0) +   # z≈3 → 1.0 근사
             weights["global"]* rwr +
             weights["temporal"]* temporal_post)
    return min(max(score, 0.0), 1.0)
