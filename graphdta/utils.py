import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        skipped = 0
        for i in range(data_len):
            if (i + 1) % 1000 == 0 or i == 0:
                print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            if smiles not in smile_graph:
                skipped += 1
                continue
            c_size, features, edge_index = smile_graph[smiles]
            if c_size == 0 or len(features) == 0:
                skipped += 1
                continue
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            if len(edge_index) == 0:
                # 无键分子（如单原子），edge_index 设为 [0, 0] 占位
                edge_tensor = torch.zeros((2, 1), dtype=torch.long)
            else:
                edge_tensor = torch.LongTensor(edge_index).transpose(1, 0)
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=edge_tensor,
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)
        if skipped > 0:
            print('Warning: skipped {} invalid SMILES / empty molecules'.format(skipped))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci_torch(y, f, max_samples=5000):
    """
    GPU 版 CI (Concordance Index) - 采样版
    使用矩阵运算加速，避免 Python 循环
    """
    y = y.view(-1).float()
    f = f.view(-1).float()
    n = len(y)
    
    if n > max_samples:
        idx = torch.randperm(n)[:max_samples]
        y = y[idx]
        f = f[idx]
        n = max_samples
    
    # 排序
    y_sorted, sort_idx = torch.sort(y)
    f_sorted = f[sort_idx]
    
    # 构建两两比较矩阵 (利用广播机制)
    # y_i > y_j 的对
    y_i = y_sorted.unsqueeze(1)  # (n, 1)
    y_j = y_sorted.unsqueeze(0)  # (1, n)
    mask = y_i > y_j  # (n, n) 下三角矩阵
    
    # 计算 z (所有可比较的对数)
    z = torch.sum(mask).float()
    
    if z == 0:
        return 0.0
    
    # 计算 S
    f_i = f_sorted.unsqueeze(1)  # (n, 1)
    f_j = f_sorted.unsqueeze(0)  # (1, n)
    
    # f_i > f_j 的对数
    S = torch.sum(mask * (f_i > f_j).float())
    # f_i == f_j 的对数 (计 0.5)
    S += 0.5 * torch.sum(mask * (f_i == f_j).float())
    
    return (S / z).item()


def ci(y, f, max_samples=5000):
    """
    Concordance Index (CI) - 采样版
    大数据集只采样 max_samples 个样本计算，避免 O(n²) 性能问题
    """
    y = np.asarray(y).flatten()
    f = np.asarray(f).flatten()
    n = len(y)
    
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        y = y[idx]
        f = f[idx]
        n = max_samples
    
    ind = np.argsort(y)
    y_sorted = y[ind]
    f_sorted = f[ind]
    
    S = 0.0
    z = 0.0
    
    for i in range(1, n):
        y_i = y_sorted[i]
        f_i = f_sorted[i]
        mask = y_sorted[:i] < y_i
        z += np.sum(mask)
        if np.any(mask):
            f_j = f_sorted[:i][mask]
            S += np.sum(f_i > f_j) + 0.5 * np.sum(f_i == f_j)
    
    return S / z if z > 0 else 0.0


# ==========================
# GPU 版指标函数（torch tensors）
# 输入: 1D torch.Tensor，在 GPU 上
# 输出: float
# ==========================
def rmse_torch(y, f):
    return torch.sqrt(torch.mean((y - f) ** 2)).item()

def mse_torch(y, f):
    return torch.mean((y - f) ** 2).item()

def pearson_torch(y, f):
    y_mean = torch.mean(y)
    f_mean = torch.mean(f)
    y_centered = y - y_mean
    f_centered = f - f_mean
    numerator = torch.sum(y_centered * f_centered)
    denominator = torch.sqrt(torch.sum(y_centered ** 2) * torch.sum(f_centered ** 2))
    if denominator == 0:
        return 0.0
    return (numerator / denominator).item()

def spearman_torch(y, f):
    """
    GPU 版 Spearman 相关系数
    先转换到 CPU 用 scipy 计算（更稳定可靠），避免 GPU 排序问题
    """
    y_cpu = y.detach().cpu().float()
    f_cpu = f.detach().cpu().float()
    rs = stats.spearmanr(y_cpu.numpy(), f_cpu.numpy())[0]
    return float(rs) if not np.isnan(rs) else 0.0


def mae_torch(y, f):
    """GPU 版 MAE"""
    return torch.mean(torch.abs(y - f)).item()


def r2_torch(y, f):
    """GPU 版 R²"""
    ss_tot = torch.sum((y - torch.mean(y)) ** 2)
    ss_res = torch.sum((y - f) ** 2)
    if ss_tot == 0:
        return 0.0
    return (1 - ss_res / ss_tot).item()


def compute_ef_torch(y_true, y_pred, top_percent=1):
    """GPU 版富集因子"""
    n_total = len(y_true)
    if n_total < 5:
        return 0.0
    
    n_top = max(1, int(n_total * top_percent / 100))
    n_active = max(1, int(n_total * 0.2))
    
    sorted_true, _ = torch.sort(y_true, descending=True)
    active_threshold = sorted_true[min(n_active - 1, len(sorted_true) - 1)]
    n_active_total = torch.sum(y_true >= active_threshold).float()
    
    if n_active_total == 0:
        return 0.0
    
    _, sorted_indices = torch.sort(y_pred, descending=True)
    y_true_sorted = y_true[sorted_indices]
    n_active_top = torch.sum(y_true_sorted[:n_top] >= active_threshold).float()
    
    ef = (n_active_top / n_top) / (n_active_total / n_total)
    return ef.item()


def compute_ece_torch(y_true, y_pred, n_bins=10):
    """GPU 版期望校准误差"""
    n_total = len(y_true)
    if n_total < 10:
        return 0.0
    
    y_min = torch.min(torch.min(y_true), torch.min(y_pred))
    y_max = torch.max(torch.max(y_true), torch.max(y_pred))
    if y_max == y_min:
        return 0.0
    
    bin_boundaries = torch.linspace(y_min, y_max, n_bins + 1, device=y_true.device)
    ece = torch.tensor(0.0, device=y_true.device)
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (y_pred >= lower) & (y_pred <= upper)
        else:
            mask = (y_pred >= lower) & (y_pred < upper)
        
        n_in_bin = torch.sum(mask).float()
        if n_in_bin > 0:
            avg_pred = torch.mean(y_pred[mask])
            avg_true = torch.mean(y_true[mask])
            ece += torch.abs(avg_pred - avg_true) * n_in_bin / n_total
    
    return ece.item()


def compute_aupr_torch(y_true, y_pred):
    """GPU 版 AUPR"""
    n_total = len(y_true)
    if n_total < 5:
        return 0.0
    
    n_pos = max(1, int(n_total * 0.2))
    sorted_true_desc, _ = torch.sort(y_true, descending=True)
    threshold = sorted_true_desc[min(n_pos - 1, len(sorted_true_desc) - 1)]
    is_positive = (y_true >= threshold).float()
    n_total_positive = torch.sum(is_positive)
    
    if n_total_positive == 0:
        return 0.0
    
    _, sorted_indices = torch.sort(y_pred, descending=True)
    sorted_positives = is_positive[sorted_indices]
    
    tp = torch.cumsum(sorted_positives, dim=0)
    fp = torch.cumsum(1 - sorted_positives, dim=0)
    precisions = tp / (tp + fp + 1e-10)
    recalls = tp / n_total_positive
    
    recalls_prev = torch.zeros_like(recalls)
    recalls_prev[1:] = recalls[:-1]
    
    aupr_val = torch.sum((recalls - recalls_prev) * precisions)
    return aupr_val.item()


def compute_all_metrics_gpu(y_true, y_pred):
    """
    纯 GPU 版指标计算
    y_true, y_pred: torch.Tensor (GPU)
    除了 Spearman (scipy 更稳定) 外，所有计算都在 GPU 上完成
    """
    y_true = y_true.view(-1).float()
    y_pred = y_pred.view(-1).float()
    
    # 基础指标 (GPU)
    rmse_val = rmse_torch(y_true, y_pred)
    mse_val = mse_torch(y_true, y_pred)
    mae_val = mae_torch(y_true, y_pred)
    r2_val = r2_torch(y_true, y_pred)
    
    # Pearson (GPU)
    pearson_val = pearson_torch(y_true, y_pred)
    
    # CI (GPU)
    ci_val = ci_torch(y_true, y_pred)
    
    # 富集因子 (GPU)
    ef1 = compute_ef_torch(y_true, y_pred, 1)
    ef5 = compute_ef_torch(y_true, y_pred, 5)
    ef10 = compute_ef_torch(y_true, y_pred, 10)
    
    # ECE (GPU)
    ece_val = compute_ece_torch(y_true, y_pred)
    
    # AUPR (GPU)
    aupr_val = compute_aupr_torch(y_true, y_pred)
    
    # Spearman 需要 CPU (scipy 更稳定可靠)
    spearman_val = spearman_torch(y_true, y_pred)
    
    return {
        'RMSE': rmse_val,
        'MSE': mse_val,
        'MAE': mae_val,
        'R2': r2_val,
        'Pearson': pearson_val,
        'Spearman': spearman_val,
        'CI': ci_val,
        'EF@1%': ef1,
        'EF@5%': ef5,
        'EF@10%': ef10,
        'ECE': ece_val,
        'AUPR': aupr_val,
    }


# ==========================
# 高级指标函数
# ==========================

def r2_score(y_true, y_pred):
    """R² 决定系数"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def mae_score(y_true, y_pred):
    """MAE 平均绝对误差"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_ef(y_true, y_pred, top_percent=1):
    """
    富集因子 (Enrichment Factor)
    top_percent: 1, 5, 10
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n_total = len(y_true)
    
    if n_total < 5:
        return 0.0
    
    n_top = max(1, int(n_total * top_percent / 100))
    n_active = max(1, int(n_total * 0.2))
    
    sorted_true = np.sort(y_true)[::-1]
    active_threshold = sorted_true[min(n_active - 1, len(sorted_true) - 1)]
    n_active_total = np.sum(y_true >= active_threshold)
    
    if n_active_total == 0:
        return 0.0
    
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    n_active_top = np.sum(y_true_sorted[:n_top] >= active_threshold)
    
    ef = (n_active_top / n_top) / (n_active_total / n_total)
    return float(ef)


def compute_ece(y_true, y_pred, n_bins=10):
    """
    期望校准误差 (Expected Calibration Error)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n_total = len(y_true)
    
    if n_total < 10:
        return 0.0
    
    y_min, y_max = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    if y_max == y_min:
        return 0.0
    
    bin_boundaries = np.linspace(y_min, y_max, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (y_pred >= lower) & (y_pred <= upper)
        else:
            mask = (y_pred >= lower) & (y_pred < upper)
        
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            avg_pred = np.mean(y_pred[mask])
            avg_true = np.mean(y_true[mask])
            ece += np.abs(avg_pred - avg_true) * n_in_bin / n_total
    
    return float(ece)


def compute_aupr(y_true, y_pred):
    """
    AUPR (Area Under Precision-Recall curve)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n_total = len(y_true)
    
    if n_total < 5:
        return 0.0
    
    n_pos = max(1, int(n_total * 0.2))
    sorted_true_desc = np.sort(y_true)[::-1]
    threshold = sorted_true_desc[min(n_pos - 1, len(sorted_true_desc) - 1)]
    is_positive = (y_true >= threshold).astype(int)
    n_total_positive = np.sum(is_positive)
    
    if n_total_positive == 0:
        return 0.0
    
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_positives = is_positive[sorted_indices]
    
    tp = np.cumsum(sorted_positives)
    fp = np.cumsum(1 - sorted_positives)
    precisions = tp / (tp + fp + 1e-10)
    recalls = tp / n_total_positive
    
    recalls_prev = np.zeros_like(recalls)
    recalls_prev[1:] = recalls[:-1]
    
    aupr_val = float(np.sum((recalls - recalls_prev) * precisions))
    return aupr_val


def compute_all_metrics(y_true, y_pred):
    """
    计算所有评估指标
    y_true, y_pred: numpy arrays
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # 基础指标
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mse_val = float(np.mean((y_true - y_pred) ** 2))
    mae_val = mae_score(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    
    # 相关性
    pearson_val = float(stats.pearsonr(y_true, y_pred)[0]) if len(y_true) > 2 else 0.0
    spearman_val = float(stats.spearmanr(y_true, y_pred)[0]) if len(y_true) > 2 else 0.0
    
    # CI
    ci_val = float(ci(y_true, y_pred))
    
    # 富集因子
    ef1 = compute_ef(y_true, y_pred, 1)
    ef5 = compute_ef(y_true, y_pred, 5)
    ef10 = compute_ef(y_true, y_pred, 10)
    
    # 校准度
    ece_val = compute_ece(y_true, y_pred)
    
    # AUPR
    aupr_val = compute_aupr(y_true, y_pred)
    
    return {
        'RMSE': rmse_val,
        'MSE': mse_val,
        'MAE': mae_val,
        'R2': r2_val,
        'Pearson': pearson_val,
        'Spearman': spearman_val,
        'CI': ci_val,
        'EF@1%': ef1,
        'EF@5%': ef5,
        'EF@10%': ef10,
        'ECE': ece_val,
        'AUPR': aupr_val,
    }