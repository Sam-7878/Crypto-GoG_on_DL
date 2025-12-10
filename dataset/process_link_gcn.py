import pandas as pd
import torch
import pickle   ## pytorch 2.6 compatibility
import numpy as np
from torch.utils.data import Dataset
from torch_scatter import scatter_add
from torch_geometric.data import InMemoryDataset, Data
import os

from pathlib import Path
import sys
# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent
# ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.settings import CHAIN
class TransactionEdgeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 chain='polygon', split='train'):
        """
        split: 'train', 'val', 'test' 중 하나
        """
        self.chain = chain
        assert split in ('train', 'val', 'test')
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.data = None
        self.load_data()

    @property
    def processed_file_names(self):
        # train / val / test 세 개의 pt 파일을 사용
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def process(self):
        train_path = f'{self.root}/{self.chain}_train_edges.txt'
        test_path  = f'{self.root}/{self.chain}_test_edges.txt'
    
        train_df = pd.read_csv(train_path, sep=' ', header=None,
                               names=['node1', 'node2', 'label'])
        test_df  = pd.read_csv(test_path,  sep=' ', header=None,
                               names=['node1', 'node2', 'label'])
    
        # --- 1) 노드 ID 매핑 ---
        all_nodes = pd.concat(
            [train_df[col] for col in ['node1', 'node2']] +
            [test_df[col]  for col in ['node1', 'node2']]
        ).unique()
        node_mapping = {node_id: idx for idx, node_id in enumerate(all_nodes)}
        
        train_df = train_df.copy()
        test_df  = test_df.copy()

        train_df['node1'] = train_df['node1'].map(node_mapping)
        train_df['node2'] = train_df['node2'].map(node_mapping)
        test_df['node1']  = test_df['node1'].map(node_mapping)
        test_df['node2']  = test_df['node2'].map(node_mapping)

        # --- 2) train_df를 train/val로 분리 (예: 80/20) ---
        train_df_shuffled = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        val_ratio = 0.2
        val_size  = int(len(train_df_shuffled) * val_ratio)

        val_df       = train_df_shuffled.iloc[:val_size].reset_index(drop=True)
        new_train_df = train_df_shuffled.iloc[val_size:].reset_index(drop=True)

        # --- 3) node feature는 train+val+test 전체 기준으로 계산 ---
        all_for_features = pd.concat([new_train_df, val_df, test_df], ignore_index=True)
        node_features = self.prepare_node_features(all_for_features)

        # --- 4) 각각 Data 객체로 변환 ---
        train_data = self.prepare_graph_data(new_train_df, node_features)
        val_data   = self.prepare_graph_data(val_df,   node_features)
        test_data  = self.prepare_graph_data(test_df,  node_features)
    
        # --- 5) pt 파일로 저장 ---
        torch.save(train_data, self.processed_paths[0])  # train_data.pt
        torch.save(val_data,   self.processed_paths[1])  # val_data.pt
        torch.save(test_data,  self.processed_paths[2])  # test_data.pt

    def load_data(self):
        # split 값에 따라 처리할 파일 인덱스 결정
        split_to_idx = {'train': 0, 'val': 1, 'test': 2}
        idx = split_to_idx[self.split]
        data_path = self.processed_paths[idx]

        # pytorch 2.6: weights_only=True 기본값 때문에 Data 객체 로딩시 문제 → False로 명시
        self.data = torch.load(data_path, weights_only=False) if os.path.exists(data_path) else None

    def prepare_node_features(self, df):
        num_nodes = df[['node1', 'node2']].max().max() + 1
        ones = torch.ones(df.shape[0], dtype=torch.long)
        node2_indices = torch.tensor(df['node2'].values, dtype=torch.long)
        node1_indices = torch.tensor(df['node1'].values, dtype=torch.long)
    
        in_degree  = torch.zeros(num_nodes, dtype=torch.long).scatter_add_(0, node2_indices, ones)
        out_degree = torch.zeros(num_nodes, dtype=torch.long).scatter_add_(0, node1_indices, ones)
        
        return torch.stack([in_degree, out_degree, in_degree + out_degree], dim=1).float()

    def prepare_graph_data(self, df, node_features):
        # 경고를 줄이기 위해 numpy로 먼저 합친 뒤 tensor로 변환
        edge_index_np = np.vstack([df['node1'].values, df['node2'].values]).astype(np.int64)
        edge_index = torch.from_numpy(edge_index_np)
        labels = torch.tensor(df['label'].values, dtype=torch.float)
        return Data(x=node_features, edge_index=edge_index.contiguous(), y=labels)

    def __getitem__(self, idx):
        # 단일 Data 객체만 있으므로 그대로 반환
        return self.data

    def __len__(self):
        # 파일 하나당 Data 하나
        return 1


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("Using chain:", CHAIN)
    chain = CHAIN
    
    root_path = f'./GoG/edges/{chain}'

    train_data = TransactionEdgeDataset(root=root_path, chain=chain, split='train')
    val_data   = TransactionEdgeDataset(root=root_path, chain=chain, split='val')
    test_data  = TransactionEdgeDataset(root=root_path, chain=chain, split='test')

