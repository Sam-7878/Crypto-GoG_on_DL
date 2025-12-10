import torch
import numpy as np
import random
from pygod.detector import DOMINANT, DONE, GAE, AnomalyDAE, CoLA
from pygod.metric import eval_roc_auc
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
from fraud_detection.graph_of_graph.utils import hierarchical_graph_reader, GraphDatasetGenerator
from torch_geometric.data import Data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint   # <-- 추가

from pathlib import Path
import sys
import pandas as pd  # <-- 추가
from datetime import datetime  # <-- 추가
# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
# ROOT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.settings import SETTINGS, CHAIN, CHAIN_LABELS


class Args:
    def __init__(self, gpu: int = 0):
        """
        gpu: PyGOD detector의 gpu 인자용
             0  이상: 해당 index GPU 사용
            -1     : CPU 사용
        """
        self.device = gpu


def create_masks(num_nodes):
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_size = int(num_nodes * 0.8)
    val_size = int(num_nodes * 0.1)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask


def eval_roc_auc(label, score):
    roc_auc = roc_auc_score(y_true=label, y_score=score)
    if roc_auc < 0.5:
        score = [1 - s for s in score]
        roc_auc = roc_auc_score(y_true=label, y_score=score)
    return roc_auc


def run_model(detector, data, seeds):
    auc_scores = []
    ap_scores = []
    
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        detector.fit(data)

        _, score, _, _ = detector.predict(
            data,
            return_pred=True,
            return_score=True,
            return_prob=True,
            return_conf=True
        )
        
        auc_score = eval_roc_auc(data.y, score)
        ap_score = average_precision_score(
            data.y.cpu().numpy(),
            score.cpu().numpy()
        )

        auc_scores.append(auc_score)
        ap_scores.append(ap_score)

    return np.mean(auc_scores), np.std(auc_scores), np.mean(ap_scores), np.std(ap_scores)


import pickle

def inspect_file(p):
    try:
        # PyTorch 2.6 안전모드 무시하고 옛날 방식으로 로드
        obj = torch.load(p, map_location="cpu", weights_only=False)
        print("torch.load 성공 →", type(obj))
    except Exception as e_t:
        print("torch.load 실패:", e_t)
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
                print("pickle.load 성공 →", type(obj))
        except Exception as e_p:
            print("pickle.load 실패:", e_p)


# training / validation / test 단계에서 라벨 분포를 출력해 보세요
def print_label_stats(dataset, name):
    # dataset 은 torch_geometric.data.Data 객체 혹은 (x, y) 튜플이라고 가정
    if hasattr(dataset, "y"):
        y = dataset.y
    else:
        _, y = dataset
    uniq, cnt = torch.unique(y, return_counts=True)
    print(f"[{name}] label distribution: {dict(zip(uniq.tolist(), cnt.tolist()))}")


from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)

def safe_ap(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_score)


def main():
    args = Args(gpu=0)  # 또는 그냥 Args()로 두고 기본값 0 사용

    print("Using chain:", CHAIN)
    chain = CHAIN

    # -----------------------------
    # 데이터 로드 및 PyG Data 구성
    # -----------------------------
    dataset_generator = GraphDatasetGenerator(
        f'./data/features/{chain}_basic_metrics_processed.csv'
    )
    data_list = dataset_generator.get_pyg_data_list()

    # 1) feature 텐서 구성
    x = torch.cat([data.x for data in data_list], dim=0)

    # 2) 계층 그래프 로드
    hierarchical_graph = hierarchical_graph_reader(
        f'./GoG/{chain}/edges/global_edges.csv'
    )
    edge_index = torch.LongTensor(list(hierarchical_graph.edges)).t().contiguous()
    global_data = Data(x=x, edge_index=edge_index, y=dataset_generator.target)
    train_mask_full, val_mask, test_mask = create_masks(global_data.num_nodes)

    # (1) 그래프 노드 집합 가져오기
    nodes_graph = sorted(hierarchical_graph.nodes())
    num_nodes_graph = len(nodes_graph)
    print("그래프 노드 수:", num_nodes_graph)
    print("그래프 노드 예시:", nodes_graph[:10])

    # (2) feature / label을 그래프 노드에 맞게 줄이기
    x_full = global_data.x
    y_full = global_data.y

    max_node = max(nodes_graph)
    assert max_node < x_full.shape[0], (
        f"그래프 노드 인덱스 중 최대값({max_node})이 "
        f"feature 행 개수({x_full.shape[0]})보다 큽니다."
    )

    idx = torch.tensor(nodes_graph, dtype=torch.long)

    x = x_full[idx]
    y = y_full[idx]
    train_mask = train_mask_full[idx]

    print("subset 이후 x shape:", x.shape)
    print("subset 이후 y shape:", y.shape)
    print("subset 이후 train_mask shape:", train_mask.shape)

    # (3) edge_index도 동일한 노드 집합 기준으로 재인덱싱
    mapping = {old: new for new, old in enumerate(nodes_graph)}

    edge_list = []
    for u, v in hierarchical_graph.edges():
        if u in mapping and v in mapping:
            edge_list.append((mapping[u], mapping[v]))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    print("edge_index shape:", edge_index.shape)
    print("edge_index max node idx:", edge_index.max().item())
    assert edge_index.max().item() < num_nodes_graph

    global_data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )
    global_data.train_mask = train_mask

    # debugging info (원래 라벨 기준)
    print("x shape:", x.shape)
    print("num_nodes (from x):", x.shape[0])
    print("num_nodes in hierarchical_graph:", hierarchical_graph.number_of_nodes())

    tmp_edge_index = torch.LongTensor(list(hierarchical_graph.edges))
    print("edge_index max idx:", tmp_edge_index.max().item())
    print("edge_index min idx:", tmp_edge_index.min().item())

    # -----------------------------
    # 결과 저장용 준비 (폴더/타임스탬프)
    # -----------------------------
    results_dir = Path("./results/fraud_detection/graph_of_graph")
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -----------------------------
    # Hyperparameter Search
    # -----------------------------
    model_params = {
        'DOMINANT': [{'hid_dim': d, 'lr': lr, 'epoch': e}
                     for d in [16, 32, 64]
                     for lr in [0.01, 0.005, 0.1]
                     for e in [50, 100, 150]],
        'DONE': [{'hid_dim': d, 'lr': lr, 'epoch': e}
                 for d in [16, 32, 64]
                 for lr in [0.01, 0.005, 0.1]
                 for e in [50, 100, 150]],
        'GAE': [{'hid_dim': d, 'lr': lr, 'epoch': e}
                for d in [16, 32, 64]
                for lr in [0.01, 0.005, 0.1]
                for e in [50, 100, 150]],
        'AnomalyDAE': [{'hid_dim': d, 'lr': lr, 'epoch': e}
                       for d in [16, 32, 64]
                       for lr in [0.01, 0.005, 0.1]
                       for e in [50, 100, 150]],
        'CoLA': [{'hid_dim': d, 'lr': lr, 'epoch': e}
                 for d in [16, 32, 64]
                 for lr in [0.01, 0.005, 0.1]
                 for e in [50, 100, 150]]
    }

    seed_for_param_selection = 42
    best_model_params = {}
    param_results = []  # <-- 하이퍼파라미터 탐색 결과 저장용

    for model_name, param_list in model_params.items():
        for param in param_list:
            detector = eval(
                f"{model_name}(hid_dim=param['hid_dim'], "
                f"num_layers=2, epoch=param['epoch'], "
                f"lr=param['lr'], gpu=args.device)"
            )
            avg_auc, std_auc, avg_ap, std_ap = run_model(
                detector,
                global_data,
                [seed_for_param_selection]
            )

            # CSV 저장용 기록
            param_results.append({
                "chain": chain,
                "model": model_name,
                "hid_dim": param['hid_dim'],
                "lr": param['lr'],
                "epoch": param['epoch'],
                "seed": seed_for_param_selection,
                "avg_auc": float(avg_auc),
                "std_auc": float(std_auc),
                "avg_ap": float(avg_ap),
                "std_ap": float(std_ap),
            })

            if model_name not in best_model_params or \
               avg_auc > best_model_params[model_name].get('Best AUC', 0):
                best_model_params[model_name] = {
                    "Best AUC": avg_auc,
                    "AUC Std Dev": std_auc,
                    "Best AP": avg_ap,
                    "AP Std Dev": std_ap,
                    "Params": param
                }

            print(
                f'Tested {model_name} with {param}: '
                f'Avg AUC={avg_auc:.4f}, Std AUC={std_auc:.4f}, '
                f'Avg AP={avg_ap:.4f}, Std AP={std_ap:.4f}'
            )

    # 하이퍼파라미터 탐색 결과 CSV 저장
    param_csv_path = results_dir / f"gog_param_search_{chain}_{run_id}.csv"
    pd.DataFrame(param_results).to_csv(param_csv_path, index=False)
    print(f"[INFO] Hyperparameter search results saved to: {param_csv_path}")

    # -----------------------------
    # Best Param으로 최종 평가
    # -----------------------------
    seeds_for_evaluation = [42, 43, 44]
    final_results = []  # <-- 최종 평가 결과 저장용

    for model_name, stats in best_model_params.items():
        param = stats['Params']
        detector = eval(
            f"{model_name}(hid_dim=param['hid_dim'], "
            f"num_layers=2, epoch=param['epoch'], "
            f"lr=param['lr'], gpu=args.device)"
        )
        avg_auc, std_auc, avg_ap, std_ap = run_model(
            detector,
            global_data,
            seeds_for_evaluation
        )

        # CSV 저장용 기록
        final_results.append({
            "chain": chain,
            "model": model_name,
            "hid_dim": param['hid_dim'],
            "lr": param['lr'],
            "epoch": param['epoch'],
            "seeds": ",".join(map(str, seeds_for_evaluation)),
            "final_avg_auc": float(avg_auc),
            "final_std_auc": float(std_auc),
            "final_avg_ap": float(avg_ap),
            "final_std_ap": float(std_ap),
        })

        print(
            f'Final Evaluation for {model_name}: '
            f'Avg AUC={avg_auc:.4f}, Std AUC={std_auc:.4f}, '
            f'Avg AP={avg_ap:.4f}, Std AP={std_ap:.4f}'
        )

    # 최종 평가 결과 CSV 저장
    final_csv_path = results_dir / f"gog_final_eval_{chain}_{run_id}.csv"
    pd.DataFrame(final_results).to_csv(final_csv_path, index=False)
    print(f"[INFO] Final evaluation results saved to: {final_csv_path}")

    # -----------------------------
    ## PyTorch Lightning을 이용한 모델 학습 및 체크포인트 저장
    import argparse
    import pathlib, shutil, warnings
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from fraud_detection.shared.utils import load_yaml
    from fraud_detection.graph_of_graph.data_module import GoGDataModule
    from fraud_detection.shared.base_model import GoGModel
  

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="polygon")
    parser.add_argument("--ckpt_dir", default="checkpoints")  # 저장 위치 인자
    args = parser.parse_args()

    cfg = load_yaml(f"common/configs/{args.dataset}.yaml")

    # 학습 데이터 파일 검사
    inspect_file(cfg["train_data"])

    dataModule = GoGDataModule(cfg)
    # ★ 여기서 데이터셋을 실제로 로드해 줌
    dataModule.setup("fit")   # train / val 로드
    dataModule.setup("test")  # test 로드

    model = GoGModel(cfg)

    # 라벨 분포 출력
    print_label_stats(dataModule.train_dataset, "train")
    print_label_stats(dataModule.val_dataset,   "val")
    print_label_stats(dataModule.test_dataset,  "test")


    # ---------- checkpoint callback ----------
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="gog-{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb],
        logger=True,
    )
    trainer.fit(model, dataModule)

    # ---------- 테스트 ----------
    trainer.test(model, dataModule)

    # ---------- 최종 checkpoint 복사 ----------
    if ckpt_cb.best_model_path:
        dst = pathlib.Path("checkpoints") / "gog_best.ckpt"
        shutil.copy(ckpt_cb.best_model_path, dst)
        print(f"✅ Best checkpoint saved → {dst.resolve()}")


if __name__ == "__main__":
    main()
