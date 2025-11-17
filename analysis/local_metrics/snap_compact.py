# 예: igraph 어댑터
import igraph as ig

class TNGraph:
    def __init__(self):
        # SNAP의 TNGraph는 방향 그래프
        self.g = ig.Graph(directed=True)
        # name 속성으로 정수 ID를 문자열로 보관 (중복 방지/빠른 조회)
        self._has_name_attr = False

    @classmethod
    def New(cls, *args):
        """
        SNAP 호환:
          - TNGraph.New()               -> 빈 그래프
          - TNGraph.New(NNodes)         -> 노드 0..NNodes-1 미리 추가 (편의)
          - TNGraph.New(NNodes, EEdges) -> 노드만 미리 추가(EEdges는 무시; Reserve와 비슷한 의도)
        주: SNAP 원형은 인자 없는 New()가 표준이지만,
            실전 호환성을 위해 NNodes/EEdges를 받아서 초기 노드 생성까지 해줍니다.
        """
        inst = cls()
        if len(args) >= 1 and isinstance(args[0], int) and args[0] > 0:
            n_nodes = args[0]
            # igraph는 정수 ID 대신 name으로 식별하는 편이 안전
            inst.g.add_vertices(n_nodes)
            inst.g.vs["name"] = [str(i) for i in range(n_nodes)]
            inst._has_name_attr = True
        return inst

    # (옵션) SNAP의 Reserve 대체: 의미상만 맞춰둠(igraph는 실예약 없음)
    def Reserve(self, NNodes: int, EEdges: int = 0):
        # 실제 메모리 예약은 불가하므로 no-op.
        # 필요하면 미리 노드만 생성해서 유사 효과
        if NNodes and NNodes > self.GetNodes():
            need = NNodes - self.GetNodes()
            self.g.add_vertices(need)
            names = self.g.vs["name"] if "name" in self.g.vs.attributes() else []
            start = 0 if not names else max(map(int, names)) + 1
            new_names = [str(i) for i in range(start, start + need)]
            self.g.vs["name"] = (names or []) + new_names
            self._has_name_attr = True

    def _ensure_name_attr(self):
        if not self._has_name_attr:
            self.g.vs["name"] = [str(i) for i in range(self.g.vcount())]
            self._has_name_attr = True

    def AddNode(self, nid: int):
        self._ensure_name_attr()
        names = set(self.g.vs["name"])
        s = str(nid)
        if s not in names:
            self.g.add_vertex(name=s)

    def AddEdge(self, src: int, dst: int):
        self._ensure_name_attr()
        for n in (src, dst):
            if str(n) not in set(self.g.vs["name"]):
                self.g.add_vertex(name=str(n))
        self.g.add_edge(str(src), str(dst))

    def GetNodes(self) -> int:
        return self.g.vcount()

    def GetEdges(self) -> int:
        return self.g.ecount()
    # 필요 시 더 매핑…


##########################################################################
# snap_compat_metrics.py
import math
import numpy as np

try:
    import igraph as ig
except ImportError:
    ig = None


def _as_igraph(G):
    """TNGraphCompat 또는 igraph.Graph → igraph.Graph 로 통일."""
    if hasattr(G, "g"):   # TNGraphCompat 같은 래퍼
        return G.g
    return G              # 이미 igraph.Graph 라고 가정


# ---------------------------------------------------------------------
# SNAP 호환: GetBfsEffDiam(G, NTestNodes, IsDir, SrcNId=-1)
# - 의미: BFS 기반 샘플링으로 90% 유효 직경(Effective Diameter) 근사
# - 반환: float (거리의 90-percentile)
# - 참고: SNAP 기본은 무방향으로 계산하는 경우가 흔함(IsDir=False 권장)
# ---------------------------------------------------------------------
def GetBfsEffDiam(G, NTestNodes, IsDir, SrcNId=-1):
    g = _as_igraph(G)
    n = g.vcount()
    if n == 0:
        return 0.0

    # 모드 결정: SNAP의 IsDir=False면 보통 무방향으로 취급
    if IsDir:
        mode = "OUT"  # 방향 그래프에서 한쪽 방향 최단거리
    else:
        # 무방향으로 계산: igraph는 별도 그래프 복사 없이 mode="ALL"로도 충분
        mode = "ALL"

    # 샘플 소스 노드 선택
    if SrcNId is not None and SrcNId >= 0:
        # name 속성(str)로 보관했다면 그걸 인덱스로 변환
        try:
            src_idx = g.vs.find(name=str(SrcNId)).index if "name" in g.vs.attributes() else int(SrcNId)
            sources = [src_idx]
        except Exception:
            # 해당 노드가 없으면 빈 결과
            return 0.0
    else:
        k = max(1, min(int(NTestNodes), n))
        rng = np.random.default_rng(0)  # 재현성 (원하면 seed 변경)
        sources = rng.choice(np.arange(n), size=k, replace=False).tolist()

    # 최단거리 수집 (자기 자신 0 제거, 도달 불가 inf 제거)
    dists = []
    for s in sources:
        # 무가중치 최단거리
        sp = g.shortest_paths(source=s, mode=mode)[0]
        for d in sp:
            if d not in (0, math.inf) and not np.isinf(d):
                dists.append(d)

    if not dists:
        return 0.0

    # 유효 직경: 90 퍼센타일
    return float(np.percentile(np.asarray(dists, dtype=float), 90))


# ---------------------------------------------------------------------
# SNAP 호환: GetClustCf(G, k)
# - 의미: k == -1  → 그래프의 평균 클러스터링 계수(노드별 local clustering의 평균)
#         k >= 0   → 차수(degree) == k 인 노드들의 평균 클러스터링 계수
# - 주의: SNAP은 무방향 클러스터링을 쓰는 관례가 많음 → 무방향으로 계산
# - 반환: float
# ---------------------------------------------------------------------
def GetClustCf(G, k):
    g = _as_igraph(G)
    if g.vcount() == 0:
        return 0.0

    # 무방향 기준으로 로컬 클러스터링 계산
    # degree < 2 인 노드는 정의상 local CC가 0이거나 미정 → 평균에서 제외
    deg = np.array(g.degree(mode="ALL"), dtype=int)
    # mode="zero"는 deg<2에도 0을 주므로, 후처리에서 제외 조건으로 거를 수 있음
    local_cc = np.array(g.transitivity_local_undirected(mode="zero"), dtype=float)

    # 평균을 낼 때는 deg >= 2 인 노드만 사용(SNAP 관행과 정합성↑)
    valid = deg >= 2

    if k is None or int(k) < 0:
        mask = valid
    else:
        k = int(k)
        mask = (deg == k) & valid

    if not np.any(mask):
        return 0.0

    return float(np.mean(local_cc[mask]))
