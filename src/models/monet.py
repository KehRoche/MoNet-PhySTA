import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from src.base.model import BaseModel
from .PhyField import GTFNO2d
from .SpitalDif import MSKGN
from .UniMoudle import *

import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def get_low_rank_eigenvectors(adj, rank, device='cpu'):
    """
    计算归一化拉普拉斯矩阵的低秩近似 (Top-k Eigenvectors)。
    对应于 Nyström 方法的目标：获取捕捉图主要结构的 m 个模式。

    参数:
        adj: 邻接矩阵 (NumPy array 或 Scipy sparse matrix)
        rank: 近似秩 m (即上一轮代码中的 nystrom_rank)
    返回:
        U_approx: Tensor [N, m], 归一化且正交的特征向量
    """
    # 1. 确保转换为 Scipy CSR 稀疏格式 (内存高效)
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    N = adj.shape[0]

    # 2. 构建归一化邻接矩阵: A_norm = D^-0.5 * A * D^-0.5
    # 注意：通常 GCN/GFT 使用 L = I - A_norm。
    # L 的"低频"（平滑）特征值对应 A_norm 的"最大"特征值。
    # 因此我们计算 A_norm 的 Largest Magnitude (LM) 特征值，这比直接计算 L 的 Smallest 稳定得多。

    # 计算度矩阵 D
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # 对称归一化
    adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

    # 3. 使用 Lanczos 算法计算前 k 个最大特征值/特征向量
    # 复杂度: O(rank * E)，E 为边数
    # k=rank+1 是为了容错，防止特征值重合导致的收敛问题，通常多算一个更稳
    eig_vals, eig_vecs = eigsh(adj_normalized, k=rank, which='LM', tol=1e-5)

    # 4. 排序 (eigsh 返回的顺序可能未排序)
    # 我们需要特征值从大到小 (对应 Laplacian 频率从小到大/低频到高频)
    sorted_indices = np.argsort(eig_vals)[::-1]
    # eig_vals = eig_vals[sorted_indices] # 如果需要特征值可保留
    U_approx = eig_vecs[:, sorted_indices]

    # 截取严格的 rank 个 (以防 eigsh 返回少于或多于预期)
    U_approx = U_approx[:, :rank]

    # 转换为 Tensor
    U_approx = torch.from_numpy(U_approx).float().to(device)

    return U_approx

class MoNet(BaseModel):
    def __init__(self, input_dim,output_dim,model_config):
        super(MoNet, self).__init__(input_dim,output_dim)
        self.corvar_dim = max(0, self.input_dim - 4)
        self.output_dim = output_dim

        self.A = model_config['adj']
        self.eigvecs,self.eigval = compute_laplacian(self.A)
        self.num_nodes = self.A.shape[0]


        batch_size = model_config['bs']
        seq_len = model_config['seq_len']
        emd_dim = model_config['emd_dim']
        #cosl
        self.width = model_config['gfno_hidden']
        self.energy_splits = model_config['energy_splits']

        #ecc
        self.topk_edges = model_config['topk_edges']
        self.graph_layers = model_config['ecc_layers']
        self.emd_dim = emd_dim
        self.activation = nn.ReLU()

        #采样频率，交通数据为5分钟一采样，一天288而空气数据3小时一采样，一天8
        self.time_interval = 288
        #条件信息编码，node_emb,tod,dow,空气数据集则额外增加dom,moy
        num_conditon = 3
        self.fea_dim = input_dim - 2

        if self.corvar_dim > 1:
            self.time_interval = 8
            num_conditon = 5
            self.fea_dim = input_dim - 4
            self.side_encoding = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.corvar_dim * seq_len,
                    out_channels=emd_dim * seq_len,
                    kernel_size=(1, 1),
                    bias=True,
                ),
                self.activation,
                nn.Dropout(p=0.15),
                MultiLayerPerceptron(emd_dim * seq_len, emd_dim * seq_len),
            )

        self.hidden_dim = self.emd_dim//2*num_conditon+emd_dim

        #新的embeding方法
        # spatial embeddings
        self.node_emb = nn.Parameter(
            torch.empty(self.num_nodes, self.emd_dim//2))
        nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        #两种不同的时间信息编码方式，将每个时刻的特性以emd_dim向量描述，并在编码时将最近时刻的emd向量取出并替换
        self.time_in_day_emb = nn.Parameter(
            torch.empty(self.time_interval, self.emd_dim//2))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
            torch.empty(7, self.emd_dim//2))
        nn.init.xavier_uniform_(self.day_in_week_emb)
        # 假设 self.emd_dim 是你的时间嵌入维度
        self.day_in_month_emb = nn.Parameter(
            torch.empty(31, self.emd_dim // 2))  # 1~31日
        nn.init.xavier_uniform_(self.day_in_month_emb)

        self.month_in_year_emb = nn.Parameter(
            torch.empty(12, self.emd_dim // 2))  # 1~12月
        nn.init.xavier_uniform_(self.month_in_year_emb)
        #self.com_fea = nn.Parameter(seq_len,self.num_nodes,self.input_dim)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=num_conditon * seq_len, out_channels=emd_dim, kernel_size=(1, 1), bias=True)

        # ==== 修改开始 ====
        # 获取配置中的近似秩，如果未配置默认为 32 或 64
        # nystrom_rank = model_config.get('nystrom_rank', 64)
        # 
        # # 生成 U_approx [N, m]
        # # 替换了原有的 self.eigvecs, self.eigval = compute_laplacian(self.A)
        # self.U_approx = get_low_rank_eigenvectors(self.A, rank=nystrom_rank)
        # 
        # # 不需要存储 full eigenvalues，只需近似的 eigenvectors
        # self.eigvecs = None
        # self.eigval = None
        # # ==================== 模型实例化 ====================
        # from .Nystrom import NyFNO2d
        # self.Field = NyFNO2d(
        #     in_channels=1,
        #     width=self.width,
        #     N=self.num_nodes,
        #     T=seq_len,
        #     modes_t=8,
        #     nystrom_rank=nystrom_rank,
        #     num_spectral_layers=2,
        #     out_steps=12
        # )
        # self.Field.inject_basis(self.U_approx)

        #  =====
        #path
        self.Field = GTFNO2d(N=self.num_nodes,T = seq_len,input = self.emd_dim,width = self.width,energy_splits=self.energy_splits)
        # from .PhyField import GeoFNO2d
        # self.Field = GeoFNO2d(N=self.num_nodes,T = seq_len,input = self.emd_dim,width = self.width,energy_splits=self.energy_splits)
        # from .PhyField import GNO2d
        # self.Field = GNO2d(N=self.num_nodes,T = seq_len,input = self.emd_dim,width = self.width,energy_splits=self.energy_splits)

        self.SptialModule = MSKGN(len=seq_len,hidden_channels=self.hidden_dim, levels=self.graph_layers,adj_matrix=self.A,topk=self.topk_edges)

        self.output_fusion = nn.Sequential(
            #phy,sptial,side_embeding
            nn.Linear(self.hidden_dim//2+(self.emd_dim if self.corvar_dim > 1 else 0)+1, self.emd_dim),
            self.activation,
            nn.Dropout(p=0.15),
            nn.Linear(self.emd_dim, self.output_dim))
        self.res_layer = MultiLayerPerceptron(self.hidden_dim, seq_len)



    def STIDemd(self,input):
        """
        input: (B, T, N, F)
            F 包含: [value, tod, dow, (optional day, month)]
        返回:
            hidden: (B, hidden_dim, N, T)
            tem_full_emb: (B, C_tem, T)
        """

        B, T, N, F = input.shape
        device, dtype = input.device, input.dtype

        # 1. 安全检查 tod / dow 范围
        assert input[..., 1].max() < 1, f"TOD max {input[..., 1].max().item()}"
        assert input[..., 2].max() < 1, f"DOW max {input[..., 2].max().item()}"

        # 2. temporal embedding 全序列
        tod_idx = (input[..., 1] * self.time_interval).long()  # (B,T,N)
        dow_idx = (input[..., 2] * 7).long()  # (B,T,N)
        tod_full_emb = self.time_in_day_emb[tod_idx]  # (B,T,N,emd)
        dow_full_emb = self.day_in_week_emb[dow_idx]  # (B,T,N,emd)

        tem_full_list = [tod_full_emb, dow_full_emb]
        if self.corvar_dim > 1:
            assert F >= 5, f"Air-quality inputs need [value, tod, dow, dom, moy], got F={F}"
            dom_idx = torch.clamp((input[..., 3] * 31).long(), 0, 30)
            moy_idx = torch.clamp((input[..., 4] * 12).long(), 0, 11)
            dom_full_emb = self.day_in_month_emb[dom_idx]
            moy_full_emb = self.month_in_year_emb[moy_idx]
            tem_full_list += [dom_full_emb, moy_full_emb]

        # 拼接 temporal embeddings -> (B, T, N, C_tem)
        tem_full_emb = torch.cat(tem_full_list, dim=-1).permute(0, 3, 2, 1)  # (B,C_tem,N,T)
        # temporal embedding 的最后一步（取 T-1 步）
        tem_last_emb = tem_full_emb[..., -1:]  # (B, C_tem, N, 1)

        # 对节点取平均，得到全局 temporal embedding
        tem_full_emb = tem_full_emb.mean(dim=2)  # (B, C_tem, T)

        # 4. time-series embedding
        # 原始输入 value + covariates reshape
        # 保持时序展开在通道维，节点在空间维
        ts_in = input.transpose(1, 2).contiguous()  # (B, N, L, F)
        ts_in = ts_in.view(B, N, -1)  # (B, N, L*F)
        ts_in = ts_in.transpose(1, 2).unsqueeze(-1)  # (B, L*F, N, 1) ✅

        time_series_emb = self.time_series_emb_layer(ts_in) # (B,N,emd_dim,1)

        # 5. node embedding
        node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)  # (B,N,emd_dim)
        node_emb = node_emb.transpose(1, 2).unsqueeze(-1)  # (B,emd_dim,N,1)

        # concate all embeddings
        hidden = torch.cat([time_series_emb, node_emb, tem_last_emb], dim=1)  # (B, hidden_dim, N, T)
        #b,emd*4,nodes,feat,
        #b,feat,nodes,seq_len
        return hidden.view(B, self.hidden_dim,N,-1),tem_full_emb

    def forward(self, input,  label=None):
        #batch,len,nodes,feat
        batch,len,nodes,feat = input.shape
        fea = input[:,:,:,:1]
        #；空气等含协变量的数据集，此处添加-2
        time = input[:,:,:,self.fea_dim:]

        mix_X,time_feats = self.STIDemd(torch.concat([fea,time],dim=-1))
        #b,feat,nodes,len x_phy:b,n,l,f
        X_phy = self.Field(fea.transpose(1,2),self.eigvecs,self.eigval,time_feats).transpose(1,2)
        #X_phy = self.Field(fea.transpose(1, 2)).transpose(1, 2)
        #return X_phy
        X_exdif = self.SptialModule(mix_X,self.A).repeat(1,1,1,len).transpose(1,3)
        x_res = self.res_layer(mix_X)
        if self.corvar_dim > 1:
            convar = input[:,:,:,1:self.corvar_dim+1]
            x_side = self.side_encoding(convar.reshape(batch,-1,nodes,1))
            x_side = x_side.reshape(batch,len,nodes,-1)
            y_hat = self.output_fusion(torch.cat((X_phy,X_exdif,x_side),dim=-1))+self.activation(x_res)
        else:
            y_hat = self.output_fusion(torch.cat((X_phy,X_exdif),dim=-1))
        return y_hat

