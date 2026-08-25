"""
Reptile-Transformer模型架构 - 元学习分子活性预测框架

核心组件：
1. 分子特征编码器：SMILES嵌入 + Morgan指纹 + RDKit描述符
2. 蛋白质特征编码器：ESM2预训练模型
3. Transformer融合模块：交叉注意力机制
4. 预测头：回归预测
5. 对比学习：增强表示学习
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ==========================
# 基础配置 - 优化配置以提升性能
# ==========================
HIDDEN_DIM = 512  # 原始维度，保证稳定性
MORGAN_BITS = 2048
MACCS_DIM = 167  # MACCS keys维度（创新点：多尺度分子特征）
DESC_DIM = 10
ESM2_DIM = 480  # facebook/esm2_t12_35M_UR50D hidden size
NUM_HEADS = 8  # 原始注意力头数
NUM_LAYERS = 2  # 原始层数
DROPOUT = 0.1  # 原始dropout


# ==========================
# 分子特征编码器
# ==========================
class MoleculeEncoder(nn.Module):
    """分子特征编码器（创新点：多尺度分子特征融合）
    
    融合三种分子特征：
    1. Morgan指纹：局部子结构信息
    2. MACCS keys：预定义官能团模式
    3. RDKit描述符：全局理化性质
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        
        # Morgan指纹投影
        self.morgan_proj = nn.Sequential(
            nn.Linear(MORGAN_BITS, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )
        
        # MACCS keys投影（创新点：补充官能团信息）
        self.maccs_proj = nn.Sequential(
            nn.Linear(MACCS_DIM, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )
        
        # 描述符投影
        self.desc_proj = nn.Sequential(
            nn.Linear(DESC_DIM, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )
        
        # 融合层：三种特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )
    
    def forward(self, morgan_fp, maccs_fp, descriptors):
        """
        Args:
            morgan_fp: (batch, MORGAN_BITS)
            maccs_fp: (batch, MACCS_DIM)
            descriptors: (batch, DESC_DIM)
        Returns:
            mol_feat: (batch, hidden_dim)
        """
        morgan_feat = self.morgan_proj(morgan_fp)
        maccs_feat = self.maccs_proj(maccs_fp)
        desc_feat = self.desc_proj(descriptors)
        combined = torch.cat([morgan_feat, maccs_feat, desc_feat], dim=-1)
        mol_feat = self.fusion(combined)
        return mol_feat


# ==========================
# 蛋白质特征编码器（ESM2）
# ==========================
class ProteinEncoder(nn.Module):
    """蛋白质特征编码器：使用ESM2预训练模型"""
    
    def __init__(self, model_path=None, output_dim=ESM2_DIM):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        model_path = model_path or os.environ.get(
            "ESM2_MODEL", "facebook/esm2_t12_35M_UR50D"
        )
        self.output_dim = output_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            add_pooling_layer=False,
        )
        hidden_dim = self.model.config.hidden_size
        if output_dim != hidden_dim:
            raise ValueError(f"ESM-2 output_dim must be {hidden_dim}; got {output_dim}")
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 移到设备
        self.model = self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def forward(self, sequence):
        """编码蛋白质序列"""
        # 截断过长序列
        if len(sequence) > 1022:
            sequence = sequence[:1022]
        
        encoded = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = encoded.pop("special_tokens_mask").to(device)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        token_repr = self.model(**encoded).last_hidden_state
        content_mask = encoded["attention_mask"].bool() & ~special_tokens_mask.bool()
        embedding = (
            token_repr * content_mask.unsqueeze(-1)
        ).sum(dim=1) / content_mask.sum(dim=1, keepdim=True).clamp_min(1)
        
        return embedding.squeeze(0)


# ==========================
# Transformer交叉注意力融合
# ==========================
class CrossAttentionLayer(nn.Module):
    """交叉注意力层"""
    
    def __init__(self, dim=HIDDEN_DIM, num_heads=NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=DROPOUT)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, query, key, value):
        """
        Args:
            query: (batch, 1, dim)
            key: (batch, 1, dim)
            value: (batch, 1, dim)
        """
        attended, _ = self.attn(query=query, key=key, value=value)
        attended = self.norm1(query + attended)
        output = self.norm2(attended + self.ffn(attended))
        return output


class BidirectionalCrossAttention(nn.Module):
    """双向交叉注意力：药物看蛋白质 + 蛋白质看药物"""
    
    def __init__(self, dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'drug_to_protein': CrossAttentionLayer(dim, num_heads),
                'protein_to_drug': CrossAttentionLayer(dim, num_heads)
            })
            for _ in range(num_layers)
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )
    
    def forward(self, drug_feat, protein_feat):
        """
        Args:
            drug_feat: (batch, dim)
            protein_feat: (batch, dim)
        """
        # 转换为序列形式
        drug_seq = drug_feat.unsqueeze(1)  # (batch, 1, dim)
        protein_seq = protein_feat.unsqueeze(1)  # (batch, 1, dim)
        
        drug_out = drug_seq
        protein_out = protein_seq
        
        consistency_losses = []
        
        # 多层交叉注意力
        for layer in self.layers:
            # 药物看蛋白质
            drug_attended = layer['drug_to_protein'](drug_out, protein_out, protein_out)
            # 蛋白质看药物
            protein_attended = layer['protein_to_drug'](protein_out, drug_out, drug_out)
            
            drug_out = drug_attended
            protein_out = protein_attended
            
            # 每一层的一致性损失
            consistency_losses.append(F.mse_loss(
                drug_out.squeeze(1), protein_out.squeeze(1)
            ))
        
        drug_out = drug_out.squeeze(1)
        protein_out = protein_out.squeeze(1)
        
        # 融合
        combined = torch.cat([drug_out, protein_out], dim=-1)
        output = self.fusion(combined)
        
        # 平均一致性损失
        consistency_loss = torch.mean(torch.stack(consistency_losses))
        
        return output, consistency_loss


# ==========================
# 预测头
# ==========================
class PredictionHead(nn.Module):
    """回归预测头"""
    
    def __init__(self, input_dim=HIDDEN_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(input_dim // 4, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


# ==========================
# 对比学习模块
# ==========================
class ContrastiveLoss(nn.Module):
    """InfoNCE对比学习损失"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, drug_repr, protein_repr):
        """
        Args:
            drug_repr: (batch, dim)
            protein_repr: (batch, dim)
        """
        # 归一化
        drug_norm = F.normalize(drug_repr, dim=-1)
        protein_norm = F.normalize(protein_repr, dim=-1)
        
        # 相似度矩阵
        sim = torch.matmul(drug_norm, protein_norm.T) / self.temperature
        
        # InfoNCE损失
        exp_sim = torch.exp(sim)
        positive = torch.diag(exp_sim)
        denominator = exp_sim.sum(dim=1)
        
        # 防止除零
        denominator = torch.clamp(denominator, min=1e-9)
        loss = -torch.mean(torch.log(positive / denominator))
        
        return loss


# ==========================
# 完整模型
# ==========================
class ReptileTransformer(nn.Module):
    """完整的Reptile-Transformer模型
    
    创新点：
    1. 双向交叉注意力融合：药物→蛋白质 + 蛋白质→药物的双向交互
    2. 一致性损失：确保双向注意力表示一致
    3. 对比学习：增强药物-蛋白质对的表示学习
    """
    
    def __init__(self):
        super().__init__()
        
        # 分子编码器
        self.molecule_encoder = MoleculeEncoder()
        
        # 蛋白质编码器（预计算，训练时不参与梯度计算）
        self.protein_encoder = None
        
        # 蛋白质特征投影（将ESM2维度对齐到HIDDEN_DIM）
        self.protein_proj = nn.Sequential(
            nn.Linear(ESM2_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )
        
        # 双向交叉注意力融合
        self.cross_attention = BidirectionalCrossAttention()
        
        # 预测头
        self.predictor = PredictionHead()
        
        # 对比学习损失
        self.contrastive_loss = ContrastiveLoss()
        
        # 输出缩放参数
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))
        
        # 移到设备
        self.to(device)
    
    def set_protein_encoder(self, encoder):
        """设置蛋白质编码器（预计算时使用）"""
        self.protein_encoder = encoder
    
    def forward(self, morgan_fp, maccs_fp, descriptors, protein_feat):
        """
        前向传播（使用预计算的蛋白质特征）
        
        Args:
            morgan_fp: (batch, MORGAN_BITS)
            maccs_fp: (batch, MACCS_DIM) - MACCS keys指纹
            descriptors: (batch, DESC_DIM)
            protein_feat: (batch, ESM2_DIM) - 预计算的蛋白质特征
        """
        # 分子编码（多尺度特征融合）
        mol_feat = self.molecule_encoder(morgan_fp, maccs_fp, descriptors)
        
        # 蛋白质特征投影
        protein_feat_proj = self.protein_proj(protein_feat)
        
        # 交叉注意力融合
        fused_feat, consistency_loss = self.cross_attention(mol_feat, protein_feat_proj)
        
        # 预测
        preds = self.predictor(fused_feat)
        preds = preds * self.output_scale + self.output_bias
        
        # 对比学习损失
        contrastive_loss = self.contrastive_loss(mol_feat, protein_feat_proj)
        
        return preds.squeeze(-1), consistency_loss, contrastive_loss
    
    def forward_with_sequence(self, morgan_fp, maccs_fp, descriptors, sequence):
        """
        前向传播（在线计算蛋白质特征，用于验证）
        
        Args:
            morgan_fp: (batch, MORGAN_BITS)
            maccs_fp: (batch, MACCS_DIM)
            descriptors: (batch, DESC_DIM)
            sequence: 蛋白质序列字符串
        """
        # 在线计算蛋白质特征
        with torch.no_grad():
            protein_embedding = self.protein_encoder(sequence)
            protein_embedding = protein_embedding.unsqueeze(0)
        
        # 前向传播
        return self.forward(morgan_fp, maccs_fp, descriptors, protein_embedding)
    
    def get_embeddings(self, morgan_fp, maccs_fp, descriptors, protein_feat):
        """获取中间特征表示"""
        mol_feat = self.molecule_encoder(morgan_fp, maccs_fp, descriptors)
        protein_feat_proj = self.protein_proj(protein_feat)
        fused_feat, _ = self.cross_attention(mol_feat, protein_feat_proj)
        return mol_feat, protein_feat_proj, fused_feat
    
    def save(self, path):
        """保存模型（不保存ESM2编码器）"""
        state_dict = {k: v for k, v in self.state_dict().items() if 'protein_encoder' not in k}
        torch.save(state_dict, path)
    
    def load(self, path):
        """加载模型"""
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict, strict=False)
        return self


# ==========================
# Reptile内循环优化器
# ==========================
class ReptileInnerOptimizer:
    """Reptile内循环优化器"""
    
    def __init__(self, model, lr=0.001):
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    def step(self, loss):
        """执行一步内循环更新"""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
    
    def reset(self):
        """重置优化器状态"""
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)


# ==========================
# 工具函数
# ==========================
def init_weights(module):
    """初始化权重"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_params(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
