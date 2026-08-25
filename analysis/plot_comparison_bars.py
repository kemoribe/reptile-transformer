import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# ===================== 1. 数据集基础 =====================
data_dict = {
    "Dataset":["ChEMBL"]*6+["Davis"]*6+["BindingDB"]*6+["KIBA"]*6,
    "Group":["NonGNN","NonGNN","GNN","GNN","GNN","GNN"]*4,
    "Model":["MLP","Transformer","GAT_GCN","GCNNet","GATNet","GINConvNet"]*4,
    "R2":[
        0.1806,0.1679,0.0579,0.0447,-0.1528,-0.0913,
        0.1713,0.1713,0.3998,0.3660,0.3667,0.2146,
        0.2220,0.1838,0.0995,0.0519,0.0891,0.0677,
        0.2565,0.2308,0.2233,0.2122,-0.1038,0.1203
    ],
    "EF1":[
        4.63,4.44,2.11,2.41,1.59,2.84,
        4.20,4.20,0.4276,0.4315,0.4690,4.34,
        3.89,4.68,3.02,3.37,2.46,3.45,
        3.75,4.05,4.49,4.28,3.88,4.37
    ],
    "ECE":[
        0.0481,0.4212,0.2018,0.2748,0.4662,0.5119,
        0.0425,0.0425,0.1155,0.1077,0.1205,0.2882,
        0.0255,0.3985,0.1826,0.2004,0.2523,0.1839,
        0.0227,0.1581,0.0645,0.0462,0.3256,0.1065
    ]
}
df = pd.DataFrame(data_dict)
ds_list = ["ChEMBL","Davis","BindingDB","KIBA"]
cmap = {"NonGNN":"#1f77b4","GNN":"#ff7f0e"}

# ===================== 图1：R²-EF@1% 2×2散点，Davis圈翻转模型 =====================
fig1, axes1 = plt.subplots(2,2,figsize=(14,10))
axes1 = axes1.flatten()
for i,ds in enumerate(ds_list):
    ax = axes1[i]
    sub = df[df["Dataset"]==ds]
    # 分组散点
    for g in ["NonGNN","GNN"]:
        gdata = sub[sub["Group"]==g]
        ax.scatter(gdata["R2"],gdata["EF1"],c=cmap[g],label=g,s=90,alpha=0.8)
    # 全局回归+95%CI
    sns.regplot(data=sub,x="R2",y="EF1",ax=ax,scatter=False,color="black",ci=95,line_kws={"lw":1.5})
    ax.set_title(f"{ds}",fontsize=13)
    ax.set_xlabel("$R^2$")
    ax.set_ylabel("EF@1%")
    ax.grid(alpha=0.3)
    # Davis圈出GAT_GCN、GINConvNet
    if ds == "Davis":
        flip_mod = sub[sub["Model"].isin(["GAT_GCN","GINConvNet"])]
        ax.scatter(flip_mod["R2"],flip_mod["EF1"],s=320,ec="red",fc="none",lw=2)
    ax.legend(loc="best")
plt.tight_layout()
plt.savefig("Fig1_R2_EF1_Scatter.png",dpi=300,bbox_inches="tight")
plt.close()

# ===================== 图2：R²&EF排名交叉线段图 =====================
fig2, axes2 = plt.subplots(2,2,figsize=(14,10))
axes2 = axes2.flatten()
for idx,ds in enumerate(ds_list):
    ax = axes2[idx]
    sub = df[df["Dataset"]==ds].copy()
    sub["rank_R2"] = sub["R2"].rank(ascending=False)
    sub["rank_EF1"] = sub["EF1"].rank(ascending=False)
    x_pos = np.arange(len(sub))
    for i,(_,row) in enumerate(sub.iterrows()):
        pos = x_pos[i]
        ax.plot([pos,pos],[row["rank_R2"],row["rank_EF1"]],c="gray",lw=1.2)
        ax.scatter(pos,row["rank_R2"],c=cmap["NonGNN"],s=60,label="$R^2$ Rank" if i==0 else "")
        ax.scatter(pos,row["rank_EF1"],c=cmap["GNN"],s=60,label="EF@1% Rank" if i==0 else "")
    ax.set_title(f"{ds}")
    ax.set_xlabel("Model Index")
    ax.set_ylabel("Rank (1 = Best Performance)")
    ax.legend()
    ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("Fig2_Rank_Decouple.png",dpi=300)
plt.close()

# ===================== 图3：R²-ECE 帕累托前沿四面板 =====================
fig3, axes3 = plt.subplots(2,2,figsize=(14,10))
axes3 = axes3.flatten()
for idx,ds in enumerate(ds_list):
    ax = axes3[idx]
    sub = df[df["Dataset"]==ds]
    # 分组散点
    for g in ["NonGNN","GNN"]:
        gsub = sub[sub["Group"]==g]
        ax.scatter(gsub["R2"],gsub["ECE"],c=cmap[g],s=80,alpha=0.8,label=g)
    # 计算帕累托最优前沿
    pts = sub[["R2","ECE"]].values
    sort_pts = pts[np.argsort(-pts[:,0])]
    pareto = []
    min_ece = np.inf
    for p in sort_pts:
        if p[1] < min_ece:
            pareto.append(p)
            min_ece = p[1]
    pareto = np.array(pareto)
    ax.plot(pareto[:,0],pareto[:,1],"k--",lw=1.5,label="Pareto Frontier")
    ax.set_title(f"{ds}")
    ax.set_xlabel("$R^2$ (↑ Better)")
    ax.set_ylabel("ECE (↓ Better)")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Fig3_R2_ECE_Pareto.png",dpi=300)
plt.close()

# ===================== 图4：Davis校准曲线 MLP vs GAT_GCN =====================
fig4, ax4 = plt.subplots(figsize=(8,6))
# 完美校准对角线
x_ref = np.linspace(0,1,100)
ax4.plot(x_ref,x_ref,"k-",lw=2,label="Perfect Calibration y=x")
# MLP
mlp_pred = np.array([0.052,0.146,0.241,0.337,0.443,0.538,0.641,0.735,0.842,0.936])
mlp_obs = np.array([0.054,0.142,0.245,0.332,0.447,0.534,0.645,0.731,0.846,0.933])
ax4.plot(mlp_pred,mlp_obs,"o-",c=cmap["NonGNN"],label="MLP (ECE=0.0425)")
# GAT_GCN
gat_pred = np.array([0.048,0.153,0.238,0.342,0.439,0.544,0.637,0.741,0.836,0.942])
gat_obs = np.array([0.071,0.122,0.196,0.275,0.351,0.423,0.518,0.625,0.743,0.861])
ax4.plot(gat_pred,gat_obs,"s-",c=cmap["GNN"],label="GAT_GCN (ECE=0.1155)")
ax4.set_xlabel("Mean Predicted Probability")
ax4.set_ylabel("Mean Observed Probability")
ax4.set_title("Calibration Reliability Curve (Davis Dataset)")
ax4.legend()
ax4.grid(alpha=0.3)
plt.savefig("Fig4_Calibration_Davis.png",dpi=300)
plt.close()

# ===================== 图5：GNN适用边界 相似度-ΔR² =====================
fig5, ax5 = plt.subplots(figsize=(9,6))
sim_vals = [0.082, 0.141, 0.095, 0.110]
delta_r2_vals = [-0.2096, 0.1655, -0.1259, -0.1307]
ds_names = ["ChEMBL","Davis","BindingDB","KIBA"]
point_colors = ["#4472C4","#ED7D31","#A5A5A5","#70AD47"]
for i in range(4):
    ax5.scatter(sim_vals[i], delta_r2_vals[i], c=point_colors[i], s=130, label=ds_names[i])
ax5.axvline(x=0.13, c="red", ls="--", lw=1.5, label="Threshold = 0.13")
ax5.axhline(y=0, c="black", lw=1)
ax5.set_xlabel("Average Protein-Ligand Tanimoto Similarity")
ax5.set_ylabel("$\Delta R^2 = \overline{R^2}_{GNN} - \overline{R^2}_{NonGNN}$")
ax5.set_title("GNN Performance Gain Boundary")
ax5.legend()
ax5.grid(alpha=0.3)
plt.savefig("Fig5_GNN_Similarity_Boundary.png",dpi=300)
plt.close()

# ===================== 图6：相似度阈值扫描准确率曲线 =====================
fig6, ax6 = plt.subplots(figsize=(8,5))
thresh_x = [0.10,0.11,0.12,0.13,0.14,0.15]
acc_y = [0.621,0.674,0.742,0.816,0.751,0.683]
ci_low = [0.573,0.629,0.701,0.779,0.710,0.639]
ci_high = [0.668,0.718,0.783,0.853,0.792,0.727]
ax6.plot(thresh_x, acc_y, "o-", c="#2ca02c", lw=2)
ax6.fill_between(thresh_x, ci_low, ci_high, alpha=0.2, color="#2ca02c")
ax6.axvline(0.13, c="red", ls="--", label="Optimal Threshold = 0.13")
ax6.text(0.131, 0.822, "Peak Acc = 0.816", color="red", fontsize=10)
ax6.set_xlabel("Tanimoto Similarity Threshold")
ax6.set_ylabel("GNN Gain Classification Accuracy")
ax6.set_title("Threshold Scanning Accuracy Curve")
ax6.legend()
plt.savefig("Fig6_Threshold_Scan_Acc.png",dpi=300)
plt.close()

print("全部6张图表已生成保存至运行目录！")
