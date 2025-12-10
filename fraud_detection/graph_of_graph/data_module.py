# fraud_detection/graph_of_graph/data_module.py
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from fraud_detection.shared.utils import load_pickle


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
        if stage == "fit" or stage is None:
            self.train_dataset = load_pickle(self.train_path)
            self.val_dataset = load_pickle(self.val_path)
        if stage == "test" or stage == "predict":
            self.test_dataset = load_pickle(self.test_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )