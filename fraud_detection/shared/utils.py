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

def load_pickle(path: str):
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