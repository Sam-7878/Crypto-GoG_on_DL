# settings.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any, Dict, Optional
import json, os, sys
import pandas as pd

AllowedChain = Literal["bnb", "ethereum", "polygon"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "chain": "bnb",
    "paths": {"data_root": "./dataset", "logs_dir": "./logs"},
    "metrics": {"n_samples": 100, "undirected": True, "rng_seed": 0},
}

@dataclass
class Settings:
    chain: AllowedChain = "bnb"
    paths: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONFIG["paths"]))
    metrics: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CONFIG["metrics"]))
    raw: Dict[str, Any] = field(default_factory=dict)  # 전체 원본 보관(옵션)

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _validate_chain(value: Any) -> AllowedChain:
    if value not in ("bnb", "ethereum", "polygon"):
        raise ValueError(f"Invalid chain='{value}'. Must be one of: bnb | ethereum | polygon")
    return value  # type: ignore[return-value]

# def _load_json(path: Path) -> Dict[str, Any]:
#     with path.open("r", encoding="utf-8") as f:
#         return json.load(f)
# common/settings.py
import json, re

def _load_json(path):
    text = path.read_text(encoding="utf-8")
    # 1) json5가 있으면 먼저 시도
    try:
        import json5  # pip install json5
        return json5.loads(text)
    except Exception:
        pass
    # 2) 간이 JSONC 파서: 주석/트레일링 콤마 제거 후 표준 json으로 로드
    #    // line comments & /* block comments */
    text_no_comments = re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.M | re.S)
    #    제거로 남은 트레일링 콤마 처리: }, ] 바로 앞의 , 제거
    text_no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", text_no_comments)
    return json.loads(text_no_trailing_commas)


def _find_settings_path(cli_arg: Optional[str]) -> Path:
    # 우선순위: CLI --settings > ENV GOG_SETTINGS > 프로젝트 루트(settings.json) > 현재 폴더
    if cli_arg:
        p = Path(cli_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--settings file not found: {p}")
        return p
    env = os.getenv("GOG_SETTINGS")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"GOG_SETTINGS points to missing file: {p}")
        return p
    # 루트 기준 탐색
    candidates = [
        Path.cwd() / "settings.json",
        Path(__file__).resolve().parent / "settings.json",
        Path.cwd().parent / "settings.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # 없으면 기본값만 사용
    return Path("settings.json")  # 존재하지 않아도 동작은 하게 둠

def _get_chain_labels() -> pd.DataFrame:
    # --- common_node.py의 핵심 블록 교체용 스니펫 ---
    # chain_labels = pd.read_csv('./data/labels.csv').query('Chain == @chain')  # :contentReference[oaicite:5]{index=5}
    lc = pd.read_csv('./data/labels.csv')
    lc['Chain'] = lc['Chain'].astype(str)
    chain = SETTINGS.chain

    alias_map = {
        'bnb': {'bnb', 'bsc', 'binance', 'binance-smart-chain'},
        'ethereum': {'ethereum', 'eth'},
        'polygon': {'polygon', 'matic'},
    }
    keys = alias_map.get(chain.lower(), {chain.lower()})
    chain_norm = str(chain).strip().lower()
    chain_labels = lc[lc['Chain'].str.lower().isin({x.lower() for x in keys})].copy()
    return chain_labels

def load_settings(settings_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Settings:
    # CLI 파라미터 파싱(간단): --settings <path> 또는 --chain <name>를 허용
    cli_settings_path = None
    cli_chain = None
    if settings_path is None:
        argv = sys.argv[1:]
        if "--settings" in argv:
            i = argv.index("--settings")
            cli_settings_path = argv[i + 1] if i + 1 < len(argv) else None
        if "--chain" in argv:
            i = argv.index("--chain")
            cli_chain = argv[i + 1] if i + 1 < len(argv) else None

    path = _find_settings_path(settings_path or cli_settings_path)
    cfg = dict(DEFAULT_CONFIG)
    if path.exists():
        cfg = _deep_merge(cfg, _load_json(path))

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    if cli_chain:
        cfg["chain"] = cli_chain

    # 검증
    cfg["chain"] = _validate_chain(cfg.get("chain"))
    # 정규화: 경로를 절대경로화(원하면 유지할 수도 있음)
    base_dir = path.parent if path.exists() else Path.cwd()
    paths = dict(cfg.get("paths") or {})
    for k, v in list(paths.items()):
        paths[k] = str((base_dir / v).resolve())
    cfg["paths"] = paths


    s = Settings(
        chain=cfg["chain"],
        paths=cfg["paths"],
        metrics=cfg["metrics"],
        raw=cfg,
    )
    return s



# 모듈 임포트 시 자동 로드
SETTINGS = load_settings()
CHAIN: AllowedChain = SETTINGS.chain
CHAIN_LABELS = _get_chain_labels()

