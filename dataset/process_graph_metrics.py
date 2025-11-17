import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import sys
# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent
# ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.settings import SETTINGS, CHAIN, CHAIN_LABELS

def main():
    # chain = 'polygon'
    print("Using chain:", CHAIN)   
    chain = CHAIN
    chain_labels = CHAIN_LABELS

    graphs1 = pd.read_csv(f'./result/{chain}_basic_metrics.csv')
    graphs2 = pd.read_csv(f'./result/{chain}_advanced_metrics_labels.csv')
    
    features = pd.merge(graphs1, graphs2, on='Contract')
    
    chain_labels['binary_category'] = chain_labels['Category'].apply(lambda x: 1 if x == 0 else 0)
    label_dict = dict(zip(chain_labels.Contract, chain_labels.binary_category))
    
    features['label'] = features['Contract'].apply(lambda x: label_dict.get(x, 0))  # Default to 0 if not found
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    
    scaler = StandardScaler()
    columns = ['Num_nodes', 'Num_edges', 'Density', 'Assortativity', 'Reciprocity', 
               'Effective_Diameter', 'Clustering_Coefficient']
    features[columns] = scaler.fit_transform(features[columns])
    
    features.to_csv(f'./data/features/{chain}_basic_metrics_processed.csv', index=False)

if __name__ == "__main__":
    main()
