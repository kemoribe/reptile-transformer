import pandas as pd
import numpy as np
import os
import json, pickle, gc
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import TestbedDataset


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return 0, [], []

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
max_seq_len = 1000

_seq_cache = {}


def seq_cat(prot):
    if prot in _seq_cache:
        return _seq_cache[prot]
    x = np.zeros(max_seq_len, dtype=np.float32)
    for i, ch in enumerate(prot[:max_seq_len]):
        if ch in seq_dict:
            x[i] = seq_dict[ch]
    _seq_cache[prot] = x
    return x


# ============================================================
# 冷启动划分：按靶点（蛋白）划分
# ============================================================
import sys as _sys
if len(_sys.argv) > 1:
    datasets = _sys.argv[1:]
else:
    datasets = ['chembl', 'davis']

os.makedirs('data/processed', exist_ok=True)


def prepare_csv(dataset):
    print('convert data for ', dataset)
    fpath = 'data/' + dataset + '/'

    split_file = fpath + 'target_split.json'
    if not os.path.exists(split_file):
        print(f'  警告: {split_file} 不存在，跳过 {dataset}')
        return False

    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
    target_split = json.load(open(split_file))

    affinity = np.asarray(affinity)

    drugs = []
    prots = []
    for d in ligands.keys():
        mol = Chem.MolFromSmiles(ligands[d])
        if mol is None:
            drugs.append(ligands[d])
        else:
            drugs.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    for t in proteins.keys():
        prots.append(proteins[t])

    print(f'\n  dataset: {dataset}')
    print(f'  drugs: {len(drugs)}, proteins: {len(prots)}')
    print(f'  Y shape: {affinity.shape}')

    protein_keys = list(proteins.keys())
    protein_id_to_split = {}
    for idx, pid in enumerate(protein_keys):
        split_name = target_split.get(pid, 'train_set')
        protein_id_to_split[idx] = split_name

    split_prot_counts = {'train_set': 0, 'val_set': 0, 'test_set': 0}
    for s in protein_id_to_split.values():
        split_prot_counts[s] = split_prot_counts.get(s, 0) + 1
    print(f'  冷启动蛋白划分: {split_prot_counts}')

    rows, cols = np.where(np.isnan(affinity) == False)
    print(f'  非 NaN 配对总数: {len(rows)}')

    split_rows = {'train': [], 'val': [], 'test': []}
    split_cols = {'train': [], 'val': [], 'test': []}
    for r, c in zip(rows, cols):
        s = protein_id_to_split[c]
        if s == 'test_set':
            split_rows['test'].append(r)
            split_cols['test'].append(c)
        elif s == 'val_set':
            split_rows['val'].append(r)
            split_cols['val'].append(c)
        else:
            split_rows['train'].append(r)
            split_cols['train'].append(c)

    for opt in ['train', 'val', 'test']:
        n = len(split_rows[opt])
        print(f'  {opt}: {n} 配对')

    opts = ['train', 'val', 'test']
    for opt in opts:
        out_csv = 'data/' + dataset + '_' + opt + '.csv'
        with open(out_csv, 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            r_list = split_rows[opt]
            c_list = split_cols[opt]
            for pair_ind in range(len(r_list)):
                f.write(drugs[r_list[pair_ind]] + ',' + prots[c_list[pair_ind]] + ',' +
                        str(affinity[r_list[pair_ind], c_list[pair_ind]]) + '\n')
        print(f'  已保存 {out_csv} ({len(split_rows[opt])} 行)')

    del ligands, proteins, affinity, target_split, rows, cols, split_rows, split_cols
    gc.collect()
    return True


def build_smile_graph_for_dataset(dataset):
    """只为当前数据集构建 smile_graph，减少内存占用"""
    smiles_set = set()
    opts = ['train', 'val', 'test']
    for opt in opts:
        csv_file = 'data/' + dataset + '_' + opt + '.csv'
        if not os.path.exists(csv_file):
            continue
        df = pd.read_csv(csv_file, usecols=['compound_iso_smiles'])
        smiles_set.update(df['compound_iso_smiles'].values)
        del df
    print(f'  构建 {len(smiles_set)} 个 SMILES 的分子图...')
    smile_graph = {}
    count = 0
    for smile in smiles_set:
        g = smile_to_graph(smile)
        if g[0] > 0:
            smile_graph[smile] = g
        count += 1
        if count % 10000 == 0:
            print(f'    已处理 {count}/{len(smiles_set)}')
    print(f'  完成，有效分子图 {len(smile_graph)} 个')
    del smiles_set
    gc.collect()
    return smile_graph


def build_pt_files(dataset, smile_graph):
    opts = ['train', 'val', 'test']
    for opt in opts:
        csv_file = 'data/' + dataset + '_' + opt + '.csv'
        if not os.path.exists(csv_file):
            continue

        processed_data_file = 'data/processed/' + dataset + '_' + opt + '.pt'
        if os.path.isfile(processed_data_file):
            print(processed_data_file, ' is already created')
            continue

        df = pd.read_csv(csv_file)
        if len(df) == 0:
            print(f'  跳过空 CSV: {csv_file}')
            continue
        drugs = df['compound_iso_smiles'].values
        prots_raw = df['target_sequence'].values
        Y_data = df['affinity'].values.astype(np.float32)

        print(f'  编码蛋白序列 {len(prots_raw)} 条...')
        XT = np.array([seq_cat(t) for t in prots_raw], dtype=np.float32)

        print('preparing ', dataset + '_' + opt + '.pt in pytorch format!')
        data = TestbedDataset(root='data', dataset=dataset + '_' + opt, xd=drugs, xt=XT, y=Y_data,
                              smile_graph=smile_graph)
        print(processed_data_file, ' has been created')

        del df, drugs, prots_raw, Y_data, XT, data
        gc.collect()


# 主流程：逐数据集处理，避免内存峰值
for dataset in datasets:
    ok = prepare_csv(dataset)
    if not ok:
        continue
    print(f'\n--- 构建 {dataset} 的分子图 ---')
    smile_graph = build_smile_graph_for_dataset(dataset)
    print(f'\n--- 生成 {dataset} 的 .pt 文件 ---')
    build_pt_files(dataset, smile_graph)
    del smile_graph
    gc.collect()
    print(f'\n{dataset} 处理完成!\n')

print('\n全部处理完成!')
