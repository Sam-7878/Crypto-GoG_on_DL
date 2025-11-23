import torch
import numpy as np
import random
from pygod.detector import DOMINANT, DONE, GAE, AnomalyDAE, CoLA
from pygod.metric import eval_roc_auc
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
from fraud_detection.graph_of_graph.utils import hierarchical_graph_reader, GraphDatasetGenerator
from torch_geometric.data import Data

from pathlib import Path
import sys
# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent
# ROOT = Path(__file__).resolve().parent.parent
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

        _, score, _, _ = detector.predict(data, return_pred=True, return_score=True, return_prob=True, return_conf=True)
        
        auc_score = eval_roc_auc(data.y, score)
        ap_score = average_precision_score(data.y.cpu().numpy(), score.cpu().numpy())

        auc_scores.append(auc_score)
        ap_scores.append(ap_score)

    return np.mean(auc_scores), np.std(auc_scores), np.mean(ap_scores), np.std(ap_scores)


def main():
    args = Args(gpu=0)  # 또는 그냥 Args()로 두고 기본값 0 사용

    # chain = 'polygon'
    print("Using chain:", CHAIN)   
    chain = CHAIN


    dataset_generator = GraphDatasetGenerator(
        f'./data/features/{chain}_basic_metrics_processed.csv'
    )
    data_list = dataset_generator.get_pyg_data_list()

    # 1) feature 텐서 구성
    x = torch.cat([data.x for data in data_list], dim=0)

    # 2) 계층 그래프 로드
    dataset_generator = GraphDatasetGenerator(f'./data/features/{chain}_basic_metrics_processed.csv')
    data_list = dataset_generator.get_pyg_data_list()

    x = torch.cat([data.x for data in data_list], dim=0)

    hierarchical_graph = hierarchical_graph_reader(
        f'./GoG/{chain}/edges/global_edges.csv'
    )
    edge_index = torch.LongTensor(list(hierarchical_graph.edges)).t().contiguous()
    global_data = Data(x=x, edge_index=edge_index, y=dataset_generator.target)
    train_mask_full, val_mask, test_mask = create_masks(global_data.num_nodes)

    # 확인용 출력
    # (1) 그래프 노드 집합 가져오기
    nodes_graph = sorted(hierarchical_graph.nodes())
    num_nodes_graph = len(nodes_graph)
    print("그래프 노드 수:", num_nodes_graph)
    print("그래프 노드 예시:", nodes_graph[:10])

    # (2) feature / label을 그래프 노드에 맞게 줄이기
    # ---- 기존 코드 (대략) ----
    # global_data.x, global_data.y, global_data.train_mask 가 이미 만들어져 있다고 가정
    x_full = global_data.x        # shape: [14464, feat_dim]
    y_full = global_data.y        # shape: [14464]  또는 [14464, 1]
    # train_mask_full = global_data.train_mask  # shape: [14464]

    # ---- 그래프 노드 기준으로 subset + reindex ----

    nodes_graph = sorted(hierarchical_graph.nodes())
    num_nodes_graph = len(nodes_graph)

    # 그래프에 쓰이는 노드 인덱스가 feature matrix 범위를 벗어나지 않는지 먼저 확인
    max_node = max(nodes_graph)
    assert max_node < x_full.shape[0], (
        f"그래프 노드 인덱스 중 최대값({max_node})이 feature 행 개수({x_full.shape[0]})보다 큽니다."
    )

    # torch 인덱스 텐서로 변환
    idx = torch.tensor(nodes_graph, dtype=torch.long)

    # feature / label / mask를 그래프 노드에 맞게 subset
    x = x_full[idx]
    y = y_full[idx]
    train_mask = train_mask_full[idx]

    print("subset 이후 x shape:", x.shape)
    print("subset 이후 y shape:", y.shape)
    print("subset 이후 train_mask shape:", train_mask.shape)


    # (3) edge_index도 동일한 노드 집합 기준으로 재인덱싱
    # old index -> new index 매핑 사전
    mapping = {old: new for new, old in enumerate(nodes_graph)}

    # 기존 네트워크엑스 그래프에서 edge 뽑아서 edge_index 생성
    edge_list = []
    for u, v in hierarchical_graph.edges():
        if u in mapping and v in mapping:  # 혹시 모를 예외 방지
            edge_list.append((mapping[u], mapping[v]))

    # 양방향으로 쓸 거면 (u,v)와 (v,u) 둘 다 넣어도 됨
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    print("edge_index shape:", edge_index.shape)  # [2, num_edges]
    print("edge_index max node idx:", edge_index.max().item())
    assert edge_index.max().item() < num_nodes_graph

   # 이 x, y, train_mask, edge_index로 global_data를 다시 구성합니다
    global_data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )
    global_data.train_mask = train_mask
    # 필요하면 val/test mask도 같은 방식으로 subset


    # hierarchical_graph 불러오기
    # for debugging.
    print("x shape:", x.shape)  # (num_nodes, num_features)
    print("num_nodes (from x):", x.shape[0])

    print("num_nodes in hierarchical_graph:", hierarchical_graph.number_of_nodes())

    # edge_index를 만들기 전에 잠깐 임시 텐서로
    tmp_edge_index = torch.LongTensor(list(hierarchical_graph.edges))
    print("edge_index max idx:", tmp_edge_index.max().item())
    print("edge_index min idx:", tmp_edge_index.min().item())


    
    model_params = {
        'DOMINANT': [{'hid_dim': d, 'lr': lr, 'epoch': e} for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        'DONE': [{'hid_dim': d, 'lr': lr, 'epoch': e} for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        'GAE': [{'hid_dim': d, 'lr': lr, 'epoch': e} for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        'AnomalyDAE': [{'hid_dim': d, 'lr': lr, 'epoch': e} for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        'CoLA': [{'hid_dim': d, 'lr': lr, 'epoch': e} for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]]
    }

    seed_for_param_selection = 42
    best_model_params = {}
    for model_name, param_list in model_params.items():
        for param in param_list:
            detector = eval(
                f"{model_name}(hid_dim=param['hid_dim'], "
                f"num_layers=2, epoch=param['epoch'], "
                f"lr=param['lr'], gpu=args.device)"
            )
            avg_auc, std_auc, avg_ap, std_ap = run_model(detector, global_data, [seed_for_param_selection])
            if model_name not in best_model_params or avg_auc > best_model_params[model_name].get('Best AUC', 0):
                best_model_params[model_name] = {
                    "Best AUC": avg_auc,
                    "AUC Std Dev": std_auc,
                    "Best AP": avg_ap,
                    "AP Std Dev": std_ap,
                    "Params": param
                }
            print(f'Tested {model_name} with {param}: Avg AUC={avg_auc:.4f}, Std AUC={std_auc:.4f}, Avg AP={avg_ap:.4f}, Std AP={std_ap:.4f}')

    seeds_for_evaluation = [42, 43, 44]
    for model_name, stats in best_model_params.items():
        param = stats['Params']
        detector = eval(f"{model_name}(hid_dim=param['hid_dim'], num_layers=2, epoch=param['epoch'], lr=param['lr'], gpu=args.device)")
        avg_auc, std_auc, avg_ap, std_ap = run_model(detector, global_data, seeds_for_evaluation)
        print(f'Final Evaluation for {model_name}: Avg AUC={avg_auc:.4f}, Std AUC={std_auc:.4f}, Avg AP={avg_ap:.4f}, Std AP={std_ap:.4f}')

if __name__ == "__main__":
    main()
