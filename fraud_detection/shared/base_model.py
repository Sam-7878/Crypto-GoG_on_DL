# fraud_detection/shared/base_model.py
import torch, pickle, pathlib
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint   # <-- 추가
from torchmetrics import AUROC, F1Score

class GoGEncoder(nn.Module):
    """
    간소화된 GoG encoder (논문 Figure-2 의 G_i 와 G^G 부분만)
    """
    def __init__(self, node_dim=47, hidden=128, latent=64):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, latent)
        self.pool = global_mean_pool

    def forward(self, x, edge_index, batch):
        x = self.node_emb(x).relu()
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.pool(x, batch)   # [B, latent]


class GoGModel(pl.LightningModule):
    """
    GoG-on-GoG 구조 : 2-level encoder + classifier
    """
    def __init__(self, cfg, node_dim=47, hidden=128, latent=64, num_classes=2):
        super().__init__()
        self.save_hyperparameters(cfg)

        # ---------- key 경로 맞추기 ----------
        node_dim = cfg.get("node_dim") or cfg["model"]["node_dim"]
        hidden   = cfg.get("hidden")   or cfg["model"]["hidden"]
        latent   = cfg.get("latent")   or cfg["model"]["latent"]
        num_classes = cfg.get("num_classes", 2)   or cfg["model"]["num_classes"]
        self.lr = cfg.get("lr", 1e-3)   or cfg["model"]["lr"]
        self.weight_decay = cfg.get("weight_decay", 1e-5)   or cfg["model"]["weight_decay"]
        # ----------------------------------------      

        self.node_encoder = GoGEncoder(node_dim, hidden, latent)
        self.graph_encoder = GoGEncoder(latent, hidden, latent)

        # Metrics
        self.classifier = nn.Sequential(
            nn.Linear(latent, latent//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent//2, num_classes)
        )

    def forward(self, data: Data):
        # --- Level-1 : node-level graph ---
        z1 = self.node_encoder(data.x, data.edge_index, data.batch)
        # --- Level-2 : graph-of-graphs (dummy batch if needed) ---
        z2 = self.graph_encoder(z1.unsqueeze(0),  # fake single graph
                                torch.tensor([[0,0]]).T,  # fake edge
                                torch.zeros(z1.size(0), dtype=torch.long))
        return self.classifier(z2)   # [1,2] or [B,2]

    # ==================== 필수 메서드 ====================
    def training_step(self, batch, batch_idx):
        print("batch type:", type(batch))
        print("batch keys:", batch.keys)  # PyG Batch 면 .x, .edge_index, .y 등이 있음 
               
        x, y = batch.x, batch.y
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.softmax(logits, dim=1)[:, 1]
        self.train_auroc.update(preds, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_auc", self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.softmax(logits, dim=1)[:, 1]
        self.val_auroc.update(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        val_auc = self.val_auroc.compute()
        self.log("val_auc", val_auc, prog_bar=True)
        self.val_auroc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch.x, batch.y
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.softmax(logits, dim=1)[:, 1]
        self.test_auroc.update(preds, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        self.log("test_auc", self.test_auroc.compute())
        self.test_auroc.reset()


    # ==================== 옵티마이저 설정 ====================
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            # verbose=True,  # ← 이 줄을 삭제하거나 주석 처리
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auc",
            },
        }
    # ====================================================

    
    @classmethod
    def load_from_checkpoint(cls, ckpt_path, **kw):
        """
        Lightning 체크포인트여도 state_dict 만 꺼내 복원
        """
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # PL 체크포인트일 경우
        if "state_dict" in state:
            state = state["state_dict"]
            # key 이름에서 "model." prefix 제거
            state = {k.replace("model.", ""): v for k, v in state.items()}
        model = cls(**kw)
        model.load_state_dict(state, strict=False)
        return model.eval()

    @torch.no_grad()
    def predict_proba(self, data: Data):
        logits = self.forward(data)
        return torch.softmax(logits, dim=1)[:, 1]   # fraud prob