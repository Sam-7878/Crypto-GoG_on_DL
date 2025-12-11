# fraud_detection/graph_of_graph/data_module.py
import torch
import pytorch_lightning as pl
from fraud_detection.shared.utils import load_pickle
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

class GoGDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg["batch_size"]
        self.num_workers = cfg.get("num_workers", 0)

        # 경로는 configs/polygon.yaml 에서 가져옴
        self.train_path = cfg["train_data"]
        self.val_path = cfg["val_data"]
        self.test_path = cfg["test_data"]

    def setup(self, stage=None):
        train_data = load_pickle(self.train_path)   # Data
        val_data   = load_pickle(self.val_path)     # Data
        test_data  = load_pickle(self.test_path)    # Data

        # ★ 단일 Data 를 리스트로 감싸서 "샘플 1개짜리 Dataset" 으로 만든다.
        self.train_dataset = [train_data]
        self.val_dataset   = [val_data]
        self.test_dataset  = [test_data]

        print(f"type(train_dataset) = {type(self.train_dataset)}; len = {len(self.train_dataset)}")
        # print(f"type(train_dataset,[object Object],) = {type(self.train_dataset,[object Object],)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,   # [Data]
            batch_size=1,         # 우선 1로 두고, 나중에 늘려도 됨
            shuffle=False,        # 샘플이 여러 개면 True 로 변경 가능
            num_workers=0,        # 문제 해결 후에 늘려도 됨
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,     # [Data]
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,    # [Data]
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )