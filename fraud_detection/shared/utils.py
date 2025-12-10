# fraud_detection/mc_extension/utils.py
import os, random, numpy as np, torch, pathlib, pickle, logging, yaml
from datetime import datetime

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def save_pickle(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# fraud_detection/shared/utils.py
import os
import pickle
import torch
from typing import Any, Union


def load_yaml(path: Union[str, os.PathLike]) -> dict:
    """YAML 파일을 읽어 파이썬 dict로 반환합니다."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pickle(path: Union[str, os.PathLike]) -> Any:
    """
    파일 확장자에 따라 적절한 로더를 사용합니다.
    - .pt / .pth 파일: torch.load 사용
    - 그 외 파일: pickle.load 사용
    """
    # 확장자 추출 후 소문자로 변환
    file_extension = os.path.splitext(str(path))[1].lower()

    if file_extension in {".pt", ".pth"}:
        # PyTorch 저장 파일
        return torch.load(path, map_location="cpu", weights_only=False)
    else:
        # 일반 pickle 파일
        with open(path, "rb") as f:
            return pickle.load(f)



def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s (%(name)s) %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def current_ts():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")