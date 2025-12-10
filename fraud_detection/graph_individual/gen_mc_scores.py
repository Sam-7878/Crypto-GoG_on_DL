"""
gen_mc_scores.py - MC Pipeline Score Extraction Module
Extracts fraud scores from trained GoG model for Monte Carlo pipeline
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# 프로젝트 경로
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)

try:
    import dgl
    from dgl.dataloading import GraphDataLoader
except ImportError as e:
    print(f"[ERROR] DGL not installed: {e}")
    print("Install with: pip install dgl")
    sys.exit(1)


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Extract Monte Carlo scores from trained GoG model'
    )
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory path')
    parser.add_argument('--dataset', type=str, default='elliptic',
                        help='Dataset name (elliptic, dblp, etc.)')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                        help='Trained model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension of model')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--extract_scores', action='store_true',
                        help='Flag to extract scores')
    parser.add_argument('--score_output', type=str, default='mc_input_scores.csv',
                        help='Output CSV filename for scores')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save output scores')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for DataLoader')
    
    return parser.parse_args()


def evaluate_and_extract_scores(model, test_loader, device, save_path='mc_input_scores.csv'):
    """
    모델에서 테스트 데이터의 fraud score를 추출하여 CSV로 저장
    
    Parameters
    ----------
    model : nn.Module
        학습된 PyTorch 모델
    test_loader : DataLoader
        테스트 데이터 로더 (DGL GraphDataLoader)
    device : torch.device
        연산 장치 (cuda 또는 cpu)
    save_path : str
        저장할 CSV 파일 경로
        
    Returns
    -------
    pd.DataFrame
        node_id, label, score를 포함한 DataFrame
    """
    model.eval()
    
    all_node_ids = []
    all_labels = []
    all_scores = []
    
    global_node_offset = 0
    
    print("[*] Starting score extraction...")
    print(f"[*] Total batches: {len(test_loader)}")
    
    with torch.no_grad():
        for batch_idx, batch_item in enumerate(test_loader):
            
            # ============================================================
            # Step 1: 배치 데이터 언패킹
            # ============================================================
            # batch_item은 다음 형태 중 하나:
            # - (graph, label): 튜플 형태
            # - graph: 그래프만
            
            if isinstance(batch_item, (tuple, list)):
                # 튜플/리스트 형태: (graph, label, ...) 
                batch_graph = batch_item,[object Object],
                batch_label = batch_item,[object Object], if len(batch_item) > 1 else None
            else:
                # 그래프만 전달된 경우
                batch_graph = batch_item
                batch_label = None
            
            # ============================================================
            # Step 2: 그래프와 라벨을 device로 이동
            # ============================================================
            batch_graph = batch_graph.to(device)
            
            if batch_label is not None and isinstance(batch_label, torch.Tensor):
                batch_label = batch_label.to(device)
            
            # ============================================================
            # Step 3: 모델 예측 (forward pass)
            # ============================================================
            try:
                logits = model(batch_graph)
            except Exception as e:
                print(f"[ERROR] Model forward pass failed at batch {batch_idx}: {e}")
                continue
            
            # ============================================================
            # Step 4: 확률 계산 (softmax)
            # ============================================================
            probs = torch.softmax(logits, dim=-1)  # [N, num_classes]
            
            # fraud 클래스 확률 (class 1이라고 가정)
            if probs.shape,[object Object], >= 2:
                fraud_probs = probs[:, 1].detach().cpu().numpy()
            else:
                # 1-class 모델인 경우 sigmoid 사용
                fraud_probs = torch.sigmoid(logits[:, 0]).detach().cpu().numpy()
            
            # ============================================================
            # Step 5: 배치 크기 계산
            # ============================================================
            current_batch_size = logits.shape,[object Object],
            
            # ============================================================
            # Step 6: node_id 추출 또는 생성
            # ============================================================
            if hasattr(batch_graph, 'ndata') and 'node_id' in batch_graph.ndata:
                # 그래프에 node_id가 있는 경우
                node_ids = batch_graph.ndata['node_id'].detach().cpu().numpy()
            else:
                # node_id가 없으면 순차 인덱스 생성
                node_ids = np.arange(
                    global_node_offset,
                    global_node_offset + current_batch_size,
                    dtype=np.int64
                )
            
            # 전역 오프셋 업데이트
            global_node_offset = global_node_offset + current_batch_size
            
            # ============================================================
            # Step 7: 라벨 처리
            # ============================================================
            if batch_label is not None and isinstance(batch_label, torch.Tensor):
                batch_labels = batch_label.detach().cpu().numpy()
            elif hasattr(batch_graph, 'ndata') and 'label' in batch_graph.ndata:
                # 그래프 노드 데이터에서 라벨 추출
                batch_labels = batch_graph.ndata['label'].detach().cpu().numpy()
            else:
                # 라벨이 없으면 -1로 채움
                batch_labels = np.full(current_batch_size, -1, dtype=np.int64)
            
            # ============================================================
            # Step 8: 배치 데이터 누적
            # ============================================================
            all_node_ids.extend(node_ids.tolist())
            all_labels.extend(batch_labels.tolist())
            all_scores.extend(fraud_probs.tolist())
            
            # ============================================================
            # Step 9: 진행 상황 출력
            # ============================================================
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
                total_processed = len(all_node_ids)
                print(f"  [{batch_idx + 1:>4d}/{len(test_loader)}] "
                      f"Processed {total_processed:>6d} nodes")
    
    # ============================================================
    # Step 10: DataFrame 생성
    # ============================================================
    results_df = pd.DataFrame({
        'node_id': all_node_ids,
        'label': all_labels,
        'fraud_score': all_scores
    })
    
    # ============================================================
    # Step 11: 저장 디렉토리 생성 및 CSV 저장
    # ============================================================
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[*] Created directory: {save_dir}")
    
    results_df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"[✓] Scores saved to: {save_path}")
    
    # ============================================================
    # Step 12: 통계 출력
    # ============================================================
    print(f"\n[Statistics]")
    print(f"  Total samples: {len(results_df):,}")
    
    if (results_df['label'] >= 0).any():
        fraud_count = (results_df['label'] == 1).sum()
        normal_count = (results_df['label'] == 0).sum()
        unknown_count = (results_df['label'] < 0).sum()
        
        print(f"  Fraud (label=1): {fraud_count:,}")
        print(f"  Normal (label=0): {normal_count:,}")
        if unknown_count > 0:
            print(f"  Unknown (label=-1): {unknown_count:,}")
    
    print(f"  Mean fraud score: {results_df['fraud_score'].mean():.6f}")
    print(f"  Std fraud score: {results_df['fraud_score'].std():.6f}")
    print(f"  Min fraud score: {results_df['fraud_score'].min():.6f}")
    print(f"  Max fraud score: {results_df['fraud_score'].max():.6f}")
    
    return results_df


def load_model(model_path, in_dim, hidden_dim, num_layers, device):
    """
    모델 로드 또는 초기화
    
    Parameters
    ----------
    model_path : str
        모델 체크포인트 경로
    in_dim : int
        입력 특성 차원
    hidden_dim : int
        히든 차원
    num_layers : int
        GNN 레이어 수
    device : torch.device
        연산 장치
        
    Returns
    -------
    nn.Module
        로드된 또는 초기화된 모델
    """
    try:
        from fraud_detection.graph_individual.models import GoGModel
        model = GoGModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=2,
            num_layers=num_layers
        )
        print("[*] GoGModel initialized")
    except ImportError:
        print("[!] GoGModel import failed, using SimpleGNN")
        model = SimpleGNN(in_dim, hidden_dim, 2, num_layers)
    
    model = model.to(device)
    
    if os.path.exists(model_path):
        print(f"[*] Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("[✓] Model loaded successfully")
    else:
        print(f"[!] Model file not found: {model_path}")
        print("[!] Proceeding with untrained model")
    
    return model


class SimpleGNN(nn.Module):
    """간단한 GNN 모델 (대체용)"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super(SimpleGNN, self).__init__()
        
        layers = []
        prev_dim = in_dim
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, g):
        if hasattr(g, 'ndata'):
            if 'feat' in g.ndata:
                h = g.ndata['feat']
            elif 'x' in g.ndata:
                h = g.ndata['x']
            else:
                raise ValueError("Graph has no node features ('feat' or 'x')")
        else:
            raise ValueError("Input must be a DGL graph with node data")
        
        return self.net(h)


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    # ============================================================
    # 1. Device 설정
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")
    
    # ============================================================
    # 2. 데이터셋 로드
    # ============================================================
    print(f"[*] Loading dataset from: {args.data_dir}")
    
    try:
        from fraud_detection.graph_individual.datasets import CryptoDataset
        dataset = CryptoDataset(root=args.data_dir, name=args.dataset)
        in_dim = dataset.num_features
        print(f"[✓] Dataset loaded: {args.dataset}")
        print(f"    Input dimension: {in_dim}")
    except Exception as e:
        print(f"[!] Dataset loading failed: {e}")
        print("[!] Using dummy dataset for testing")
        
        # 테스트용 더미 데이터
        num_nodes = 1000
        in_dim = 64
        num_edges = 5000
        
        src = np.random.randint(0, num_nodes, num_edges)
        dst = np.random.randint(0, num_nodes, num_edges)
        
        g = dgl.graph((src, dst))
        g.ndata['feat'] = torch.randn(num_nodes, in_dim)
        g.ndata['label'] = torch.randint(0, 2, (num_nodes,))
        
        dataset = [g]
    
    # ============================================================
    # 3. DataLoader 생성
    # ============================================================
    print(f"[*] Creating DataLoader with batch_size={args.batch_size}")
    
    test_loader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    
    print(f"[✓] DataLoader created: {len(test_loader)} batches")
    
    # ============================================================
    # 4. 모델 로드/초기화
    # ============================================================
    print(f"[*] Initializing model...")
    model = load_model(
        args.model_path,
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device
    )
    
    # ============================================================
    # 5. Score 추출
    # ============================================================
    if args.extract_scores:
        print(f"\n[*] Extracting scores...")
        
        score_path = os.path.join(args.save_dir, args.score_output)
        
        results_df = evaluate_and_extract_scores(
            model,
            test_loader,
            device,
            save_path=score_path
        )
        
        print(f"\n[✓] Score extraction completed!")
        print(f"[*] Next step: Use '{score_path}' for MC pipeline")
    else:
        print("[!] --extract_scores flag not set")
        print("[*] Run with --extract_scores to extract scores")


if __name__ == '__main__':
    main()