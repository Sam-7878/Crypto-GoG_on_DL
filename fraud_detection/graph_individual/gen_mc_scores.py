"""
gen_mc_scores.py
MC 파이프라인용 Score 추출 스크립트
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 프로젝트 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from dgl.dataloading import GraphDataLoader
    import dgl
except ImportError:
    print("DGL not installed. Run: pip install dgl")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract MC scores from trained model')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='elliptic')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--extract_scores', action='store_true')
    parser.add_argument('--score_output', type=str, default='mc_input_scores.csv')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    return parser.parse_args()


def evaluate_and_extract_scores(model, test_loader, device, save_path='mc_input_scores.csv'):
    """
    테스트 데이터에서 fraud score를 추출하여 CSV로 저장
    """
    model.eval()

    all_node_ids = []
    all_labels = []
    all_scores = []
    global_offset = 0

    print("Processing batches...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):

            # 배치 데이터 언패킹
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) >= 2:
                    graphs = batch_data,[object Object],
                    labels = batch_data,[object Object],
                else:
                    graphs = batch_data,[object Object],
                    labels = None
            else:
                graphs = batch_data
                labels = None

            # 그래프에서 라벨 추출 시도
            if labels is None:
                if hasattr(graphs, 'ndata') and 'label' in graphs.ndata:
                    labels = graphs.ndata['label']

            # 디바이스로 이동
            graphs = graphs.to(device)
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)

            # 모델 예측
            logits = model(graphs)
            probs = torch.softmax(logits, dim=-1)

            # fraud 확률 (class 1)
            fraud_scores = probs[:, 1].detach().cpu().numpy()

            # 배치 크기
            batch_size = logits.shape,[object Object],

            # node_id 처리
            if hasattr(graphs, 'ndata') and 'node_id' in graphs.ndata:
                node_ids = graphs.ndata['node_id'].detach().cpu().numpy()
            else:
                node_ids = np.arange(global_offset, global_offset + batch_size)

            global_offset = global_offset + batch_size

            # label 처리
            if isinstance(labels, torch.Tensor):
                batch_labels = labels.detach().cpu().numpy()
            else:
                batch_labels = np.full(batch_size, -1, dtype=np.int64)

            # 누적
            all_node_ids.extend(node_ids.tolist())
            all_labels.extend(batch_labels.tolist())
            all_scores.extend(fraud_scores.tolist())

            # 진행률 출력
            if (batch_idx + 1) % 50 == 0:
                print("  Batch {}/{} done ({} samples)".format(
                    batch_idx + 1, len(test_loader), len(all_node_ids)))

    # DataFrame 생성
    df = pd.DataFrame({
        'node_id': all_node_ids,
        'label': all_labels,
        'score': all_scores
    })

    # 저장
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    df.to_csv(save_path, index=False)

    print("")
    print("Score extraction completed!")
    print("  Saved to: {}".format(save_path))
    print("  Total samples: {}".format(len(df)))

    if (df['label'] >= 0).any():
        fraud_count = (df['label'] == 1).sum()
        normal_count = (df['label'] == 0).sum()
        print("  Fraud (label=1): {}".format(fraud_count))
        print("  Normal (label=0): {}".format(normal_count))

    return df


class SimpleGNNModel(nn.Module):
    """간단한 GNN 모델 (실제 모델로 교체 필요)"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super(SimpleGNNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.relu = nn.ReLU()

    def forward(self, g):
        if hasattr(g, 'ndata') and 'feat' in g.ndata:
            h = g.ndata['feat']
        else:
            h = g.ndata['x'] if 'x' in g.ndata else None
            if h is None:
                raise ValueError("Graph has no node features")

        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h)
            h = self.relu(h)

        h = self.layers[-1](h)
        return h


def main():
    args = parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))

    # 데이터셋 로드 (프로젝트에 맞게 수정 필요)
    print("Loading dataset...")

    try:
        from fraud_detection.graph_individual.datasets import CryptoDataset
        dataset = CryptoDataset(root=args.data_dir, name=args.dataset)
        in_dim = dataset.num_features
    except Exception as e:
        print("Dataset loading failed: {}".format(e))
        print("Using dummy data for testing...")

        # 테스트용 더미 데이터
        num_nodes = 100
        in_dim = 64
        g = dgl.graph((np.random.randint(0, num_nodes, 200),
                       np.random.randint(0, num_nodes, 200)))
        g.ndata['feat'] = torch.randn(num_nodes, in_dim)
        g.ndata['label'] = torch.randint(0, 2, (num_nodes,))
        dataset = [g]

    # Test loader
    test_loader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    print("Test loader ready: {} batches".format(len(test_loader)))

    # 모델 초기화
    try:
        from fraud_detection.graph_individual.models import GoGModel
        model = GoGModel(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            out_dim=2,
            num_layers=args.num_layers
        )
    except Exception as e:
        print("GoGModel loading failed: {}".format(e))
        print("Using SimpleGNNModel...")
        model = SimpleGNNModel(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            out_dim=2,
            num_layers=args.num_layers
        )

    model = model.to(device)

    # 체크포인트 로드
    if os.path.exists(args.model_path):
        print("Loading model from {}".format(args.model_path))
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("Model file not found: {}".format(args.model_path))
        print("Proceeding with untrained model...")

    # Score 추출
    if args.extract_scores:
        print("")
        print("Starting score extraction...")
        score_path = os.path.join(args.save_dir, args.score_output)
        evaluate_and_extract_scores(model, test_loader, device, save_path=score_path)
        print("")
        print("Done! Output file: {}".format(score_path))


if __name__ == '__main__':
    main()