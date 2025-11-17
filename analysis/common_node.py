import pandas as pd
import json
from tqdm import tqdm
import os 

from pathlib import Path
import sys
# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent
# ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from common.settings import SETTINGS, CHAIN

print("Using chain:", CHAIN)   
chain = CHAIN

# --- common_node.py의 핵심 블록 교체용 스니펫 ---
# chain_labels = pd.read_csv('./data/labels.csv').query('Chain == @chain')  # :contentReference[oaicite:5]{index=5}
lc = pd.read_csv('./data/labels.csv')
lc['Chain'] = lc['Chain'].astype(str)

alias_map = {
    'bnb': {'bnb', 'bsc', 'binance', 'binance-smart-chain'},
    'ethereum': {'ethereum', 'eth'},
    'polygon': {'polygon', 'matic'},
}
keys = alias_map.get(chain.lower(), {chain.lower()})
chain_labels = lc[lc['Chain'].str.lower().isin({x.lower() for x in keys})].copy()


chain_norm = str(chain).strip().lower()
chain_class = list(chain_labels.Contract.values)

output_file = f'./graphs/{chain}/{chain}_common_nodes_except_null_labels.csv'  # :contentReference[oaicite:6]{index=6}
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 1) timestamp 표준화
    if 'timestamp' in df.columns:
        ts = df['timestamp']
    elif 'timeStamp' in df.columns:
        ts = df['timeStamp']
    elif 'block_timestamp' in df.columns:
        ts = df['block_timestamp']
    else:
        raise KeyError("No timestamp column among ['timestamp','timeStamp','block_timestamp']")
    # 초/밀리초 자동 판별
    ts_int = pd.to_numeric(ts, errors='coerce')
    if ts_int.notna().any():
        # 값이 1e12 이상이면 ms로 가정
        unit = 'ms' if ts_int.max() > 1_000_000_000_000 else 's'
        df['__ts_std__'] = pd.to_datetime(ts_int, unit=unit, errors='coerce')
    else:
        # 문자열 형식 날짜일 수도 있음
        df['__ts_std__'] = pd.to_datetime(ts, errors='coerce')

    # 2) from/to 표준화
    if 'from' not in df.columns:
        for cand in ['from_address', 'sender', 'src', 'fromAddr']:
            if cand in df.columns:
                df['from'] = df[cand]
                break
    if 'to' not in df.columns:
        for cand in ['to_address', 'recipient', 'dst', 'toAddr']:
            if cand in df.columns:
                df['to'] = df[cand]
                break
    if ('from' not in df.columns) or ('to' not in df.columns):
        raise KeyError("No from/to columns (tried: from|from_address|sender|src|fromAddr, to|to_address|recipient|dst|toAddr)")

    return df

cutoff_date = pd.Timestamp('2024-03-01')  # :contentReference[oaicite:7]{index=7}

with open(output_file, 'w', newline='') as csvfile:
    csvfile.write('Contract1,Contract2,Common_Nodes,Unique_Addresses\n')

    errors = []
    contract_addresses = {}

    for addr in tqdm(chain_class):
        try:
            file_path = f'./data/transactions/{chain}/{addr}.csv'  # :contentReference[oaicite:8]{index=8}
            df = pd.read_csv(file_path)
            df = _normalize_columns(df)
            tx = df[df['__ts_std__'] < cutoff_date]

            addresses = pd.concat([tx['from'], tx['to']], ignore_index=True).dropna().unique()
            contract_addresses[addr] = set(map(str.lower, addresses))  # 주소 표준화(소문자)
        except Exception as e:
            errors.append((addr, str(e)))
            # print 유지 추천
            print(f'Error with address {addr}: {e}')

    # 페어 구성은 실제로 로드에 성공한 주소만(빈 딕셔너리로 인한 KeyError 방지)
    present = [a for a in chain_class if a in contract_addresses]

    null_addresses = {'0x0000000000000000000000000000000000000000'}  # :contentReference[oaicite:9]{index=9}

    from itertools import combinations
    with open(output_file, 'a', newline='') as csvfile:
        for con1, con2 in tqdm(list(combinations(present, 2))):
            s1 = contract_addresses[con1] - null_addresses
            s2 = contract_addresses[con2] - null_addresses
            common_nodes = len(s1 & s2)
            unique_addresses = len(s1 | s2)
            csvfile.write(f"{con1},{con2},{common_nodes},{unique_addresses}\n")

# Futher create the global graph based on threshold.
# threshold = 1
# global_graph = pd.read_csv(output_file)
# global_graph['Jaccard_Coefficient'] = global_graph['Common_Nodes']/global_graph['Unique_Addresses']
# global_graph = global_graph.query('Jaccard_Coefficient > @threshold')
# global_graph.to_csv(f'../data/global_graph/{chain}_graph_more_than_{threshold}_ratio.csv', index = 0) 