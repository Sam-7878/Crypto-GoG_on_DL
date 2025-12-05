import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
import dgl
from dgl.dataloading import GraphDataLoader

# 기존 모델 및 데이터셋 import (프로젝트 구조에 맞게 조정)
from models import GoGModel  # 실제 모델 클래스명으로 변경
from datasets import CryptoDataset  # 실제 데이터셋 클래스명으로 변경

def parse_args():
    parser = argparse.ArgumentParser(description='GoG Fraud Detection')
    
    # 데이터 관련
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--dataset', type=str, default='elliptic', help='Dataset name')
    
    # 모델 관련
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    
    # 평가 및 저장
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Model save directory')
    parser.add_argument('--extract_scores', action='store_true', help='Extract scores for MC pipeline')
    parser.add_argument('--score_output', type=str, default='mc_input_scores.csv', help='Score output file')
    
    return parser.parse_args()


def evaluate_and_extract_scores(model, test_loader, device, save_path='mc_input_scores.csv'):
    """
    모델 평가 및 MC 파이프라인용 Score 데이터 추출
    """
    model.eval()
    
    all_node_ids = []
    all_labels = []
    all_scores = []
    
    batch_offset = 0  # 전체 노드 인덱스 오프셋
    
    print("   📦 Processing batches...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            try:
                # 배치 데이터 처리 (DGL 배치 형식 호환)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    graphs, labels = batch_data,[object Object],, batch_data,[object Object],
                else:
                    graphs = batch_data
                    labels = graphs.ndata.get('label', None)
                
                graphs = graphs.to(device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                
                # 모델 예측
                logits = model(graphs)
                
                # **핵심 수정: 배치 크기 올바르게 추출**
                current_batch_size = logits.shape,[object Object],  # ✅ logits.shape,[object Object], 사용
                
                # 확률 계산 (fraud 클래스 확률)
                probs = torch.softmax(logits, dim=-1)
                positive_scores = probs[:, 1].cpu().numpy()  # 클래스 1 (fraud)
                
                # Node ID 생성
                if hasattr(graphs, 'ndata') and 'node_id' in graphs.ndata:
                    node_ids = graphs.ndata['node_id'].cpu().numpy()
                else:
                    # 순차 ID 할당 (MC 파이프라인에서 사용 가능)
                    node_ids = np.arange(batch_offset, batch_offset + current_batch_size)
                
                # 오프셋 업데이트
                batch_offset += current_batch_size
                
                # 데이터 누적
                all_node_ids.extend(node_ids.tolist())
                all_labels.extend(labels.cpu().numpy().tolist() if labels is not None else [-1] * current_batch_size)
                all_scores.extend(positive_scores.tolist())
                
                # 진행률 출력
                if (batch_idx + 1) % 50 == 0:
                    print(f"      Batch {batch_idx+1}/{len(test_loader)} ({batch_offset} samples)")
                    
            except Exception as e:
                print(f"⚠️  Batch {batch_idx} error: {e}")
                continue
    
    # CSV 저장
    df_scores = pd.DataFrame({
        'node_id': all_node_ids,
        'true_label': all_labels,
        'fraud_probability': all_scores
    })
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    df_scores.to_csv(save_path, index=False)
    
    print(f"\n✅ 완료! {save_path}")
    print(f"   📈 총 노드: {len(df_scores):,}")
    print(f"   🔴 Fraud (label=1): {(df_scores['true_label'] == 1).sum():,}")
    print(f"   🟢 Normal (label=0): {(df_scores['true_label'] == 0).sum():,}")
    print(f"   💾 평균 Fraud 확률: {df_scores['fraud_probability'].mean():.4f}")
    
    return df_scores


def train_epoch(model, train_loader, optimizer, criterion, device):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    
    for batch_data in train_loader:
        if isinstance(batch_data, tuple):
            graphs, labels = batch_data
            graphs = graphs.to(device)
            labels = labels.to(device)
        else:
            graphs = batch_data.to(device)
            labels = graphs.ndata['label']
        
        optimizer.zero_grad()
        logits = model(graphs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    """검증 세트 평가"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            if isinstance(batch_data, tuple):
                graphs, labels = batch_data
                graphs = graphs.to(device)
                labels = labels.to(device)
            else:
                graphs = batch_data.to(device)
                labels = graphs.ndata['label']
            
            logits = model(graphs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # 메트릭 계산
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc_score = roc_auc_score(all_labels, all_probs)
    
    return f1, auc_score


def main(args):
    # 1. Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # 2. 데이터셋 로드
    print(f"📂 Loading dataset from {args.data_dir}...")
    dataset = CryptoDataset(root=args.data_dir, name=args.dataset)
    
    # 3. Train/Val/Test 분할
    # 데이터셋 클래스에 split 메서드가 있다고 가정
    if hasattr(dataset, 'get_idx_split'):
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    else:
        # 수동 분할 (8:1:1 비율)
        num_samples = len(dataset)
        indices = np.random.permutation(num_samples)
        train_size = int(0.8 * num_samples)
        val_size = int(0.1 * num_samples)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
    
    # Subset 생성
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    # 4. DataLoader 생성
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )
    
    test_loader = GraphDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 순서 유지 필수!
        drop_last=False,
        num_workers=4
    )
    
    print(f"✅ Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 5. 모델 초기화
    model = GoGModel(
        in_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=2,  # Binary classification
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    print(f"🧠 Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 6. Optimizer 및 Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # 7. 학습 루프
    best_val_auc = 0
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n🚀 Starting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_f1, val_auc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        
        # Best model 저장
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"   ✨ New best model saved! (AUC: {val_auc:.4f})")
    
    # 8. Best model 로드
    print(f"\n📥 Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    
    # 9. Test 평가
    test_f1, test_auc = evaluate(model, test_loader, device)
    print(f"\n🎯 Test Results - F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    
    # 10. MC 파이프라인용 Score 추출
    if args.extract_scores:
        print("\n📊 Extracting scores for MC pipeline...")
        score_path = os.path.join(args.save_dir, args.score_output)
        evaluate_and_extract_scores(model, test_loader, device, save_path=score_path)
        print(f"✅ MC pipeline input ready at: {score_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)