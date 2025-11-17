import pandas as pd
import json
import random
import torch
import os
    
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

    # Use three-class as an example.
    n = 3
    category_counts = chain_labels['Category'].value_counts()
    select_class = list(category_counts.head(n).index)
    category_to_label = {category: i for i, category in enumerate(select_class)}
    chain_labels['Category'] = chain_labels['Category'].map(category_to_label)

    labels_select = list(chain_labels.query('Category < @n').Category.values)
    labels_select_df = chain_labels.query('Category < @n').reset_index(drop = True)

    # create timestamps
    stats = []
    for addr in labels_select_df.Contract.values:
        tx = pd.read_csv(f'./data/transactions/{chain}/{addr}.csv')
        first_timestamp = tx['timestamp'].min()
        stats.append({'address': addr, 'first_timestamp': first_timestamp})
        
    timestamps = pd.DataFrame(stats)

    # create index mapping
    all_address = list(labels_select_df.Contract.values)
    index_mapping = {addr: idx for idx, addr in enumerate(all_address)}

    timestamps['addr_index'] = timestamps['address'].apply(lambda x: index_mapping[x])
    timestamps = timestamps.sort_values(by = 'first_timestamp')

    train_num = int(len(list(timestamps['addr_index'].values)) * 0.8)
    test_num = len(list(timestamps['addr_index'].values)) - train_num

    train_index = list(timestamps['addr_index'].values)[:train_num]
    test_index = list(timestamps['addr_index'].values)[train_num:]

    file_name = f'./GoG/node/{chain}_train_index_{n}.txt'
    with open(file_name, 'w') as file:
        for item in train_index:
            file.write(f"{item}\n")

    file_name = f'./GoG/node/{chain}_test_index_{n}.txt'
    with open(file_name, 'w') as file:
        for item in test_index:
            file.write(f"{item}\n")

if __name__ == "__main__":
    main()