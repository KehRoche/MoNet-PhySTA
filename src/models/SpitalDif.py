import torch
import torch.nn as nn
from collections import defaultdict

from sympy import false


import torch
import torch.nn.functional as F
from torch import nn
import networkx as nx
import community as community_louvain
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter

class MSKGN(nn.Module):
    def __init__(self, len,hidden_channels: int, levels: int, adj_matrix, topk: int = 5):
        """
        Multi-Scale Kernel Graph Network
        Args:
            hidden_channels:      每个节点的隐藏维度 F
            levels:               金字塔层数 L
            kernel_nn_hidden:     用于每层 edge‐MLP 的中间维度
            topk:                 每层粗化后保留 Top-k 边
            gamma:                RBF 核参数
        """
        super().__init__()
        self.levels = levels
        self.topk = topk
        self.vis = false
        # down / mid / up 三组卷积列表
        self.conv_down = nn.ModuleList()
        self.conv_mid  = nn.ModuleList()
        self.conv_up   = nn.ModuleList()
        for l in range(levels):
            # 同层 mid-scale 卷积
            self.conv_mid.append(BatchedSparseFiLMConv(hidden_channels, hidden_channels, aggr='mean'))
            # 下行跨层 conv_down[l]: l->l+1

            self.conv_down.append(BatchedSparseFiLMConv(hidden_channels, hidden_channels, aggr='mean'))

            # 上行跨层 conv_up[l]: (l+1)->l
            self.conv_up.append(BatchedSparseFiLMConv(hidden_channels, hidden_channels, aggr='mean'))
        self.center,self.part,self.mg = self.multiscale_graph(adj_matrix.squeeze(-1))
        self.projs = nn.Sequential(
            nn.Linear( 4 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear( hidden_channels, hidden_channels//2)
            )
        # 输入与输出投影
        self.fc_in  = nn.Linear(1, hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, 1)

    def multiscale_graph(self, A: torch.Tensor):
        """
        构造 down/mid/up 三组边索引与权重。
        返回 dict:
          {
            'clusters':    List[Tensor[n_l]]           # 每层节点（或超节点）原图索引
            'edge_down':   List[(ei, ew)] l->l+1
            'edge_mid':    List[(ei, ew)] within-layer
            'edge_up':     List[(ei, ew)] l+1->l
          }
        """
        device = A.device
        N = A.size(0)
        # 初始图
        clusters = []
        edge_down, edge_mid, edge_up = [], [], []
        # 0 层：全部节点
        curr_idx = torch.arange(N, device=device)
        clusters.append(curr_idx)

        # 构建 NetworkX 做 Louvain
        A = A.clamp(min=0.0)
        G = nx.from_numpy_array(A.detach().cpu().numpy())
        for u,v in G.edges():
            G[u][v]['weight'] = float(A[u,v])
        part = community_louvain.best_partition(G, weight='weight')
        # 社区中心
        # 社区 -> 成员节点映射
        comms = defaultdict(list)
        for node, comm in part.items():
            comms[comm].append(node)

        # 社区编号列表
        # 找到每个社区的“中心节点” = 权重度最大节点
        centers = {comm: max(nodes, key=lambda i: G.degree(i, weight='weight'))
                   for comm, nodes in comms.items()}

        # 建立原始节点 -> 所属中心映射
        node2center = {node: centers[comm] for comm, nodes in comms.items() for node in nodes}
        # 2. 构建中心节点 -> 成员节点的映射
        center2nodes = defaultdict(list)
        for node, center in node2center.items():
            center2nodes[center].append(node)

        # Step 3: 初始化粗化图（保持原始图维度）
        A_coarse = torch.zeros_like(A)
        # Step 4: 汇聚边权到对应中心节点之间
        for u in range(A.shape[0]):
            for v in range(A.shape[1]):
                cu = node2center[u]
                cv = node2center[v]
                if cu == cv:
                    continue  # （可选）不考虑内部自连接
                A_coarse[cu, cv] += A[u, v]
        # （可选）去除自环
        A_up = torch.zeros_like(A)
        for cu in center2nodes:
            for cv in center2nodes:
                if cu == cv:
                    continue
                edge_weight = A_coarse[cu, cv]
                if edge_weight == 0:
                    continue  # 跳过无连接超节点
                for i in center2nodes[cu]:
                    for j in center2nodes[cv]:
                        A_up[i, j] += edge_weight  # 累加方式
        # For each row i, retain top_k entries, zero out others
        topk_vals, topk_idx = torch.topk(A_up, k=min(self.topk, N), dim=1)
        mask = torch.zeros_like(A_up)
        rows = torch.arange(N, device=device).unsqueeze(1).repeat(1, topk_idx.size(1))
        mask[rows, topk_idx] = 1
        A_up = A_up * mask + A
        # 记录 down & up
        ei_d, ew_d = dense_to_sparse(A_coarse.to(device))
        edge_down.append((ei_d, ew_d.unsqueeze(-1)))

        ei_u, ew_u = dense_to_sparse((A_up).to(device))
        edge_up.append((ei_u, ew_u.unsqueeze(-1)))

        ei_m, ew_m = dense_to_sparse(A.to(device))
        edge_mid.append((ei_m, ew_m.unsqueeze(-1)))

        return centers,part,{
            'clusters':  clusters,
            'edge_down': edge_down,
            'edge_mid':  edge_mid,
            'edge_up':   edge_up
        }

    def forward(self, x, adj_matrix):
        """
        x: [B, N, F] 原始时空量（如每个节点的时间序列某一时刻）
        adj_matrix: [N,N]
        """
        device = x.device
        x = x.squeeze(-1).transpose(-1, -2)
        B, N, dim = x.size()
        # 投影到 hidden
        #x = self.fc_in(x)

        # 多尺度图预处理
        mg = self.mg
        # clusters, edge_down/mid/up 各为-length lists
        features = {'input': x}
        # V-Cycle 消息传递
        for l in range(self.levels):
            ei, ew = mg['edge_down'][0]
            x_l = x.clone()  # 当前层 feature
            # 聚合到粗层节点 order 与 clusters[l+1] 对齐
            msg = self.conv_down[l](x_l, ei, ew)  # [B, n_{l+1}, F]
            # 将 msg 对应位置取出
            for node, comm in self.part.items():
                center = self.center[comm]  # 找到当前节点所属社区的中心节点
                x_l[:,node,:] = msg[:,center,:]  # 将中心节点特征映射给该节点
            features[f'down_l{l}'] = x_l
            # 同层 mid
            ei_m, ew_m = mg['edge_mid'][0]
            x_m = self.conv_mid[l](x, ei_m, ew_m)
            #x_m = F.relu(x_m)
            features[f'mid_l{l}'] = x_m

            # up-scale
            ei_u, ew_u = mg['edge_up'][0]
            x_up = self.conv_up[l](x, ei_u, ew_u)
            #x_up = F.relu(x_up)
            features[f'up_l{l}'] = x_up

            concat = torch.cat([x, x_l, x_m,x_up], dim=-1)  # [B,N,F+3H]
            x = self.projs(concat)
        if self.vis:
            # Select a batch and node to visualize
            batch_idx = 0
            first_ei = mg['edge_down'][l][0]
            # pick first source node
            node_idx = int(first_ei[0,0].item())
            import matplotlib.pyplot as plt
            # Plot feature values across stages
            stages = list(features.keys())
            plt.figure(figsize=(15,5))
            for stage in stages:
                feat_vec = features[stage][batch_idx, node_idx, :].detach().cpu().numpy()
                plt.plot(range(dim), feat_vec, label=stage)

            plt.title(f'Feature Evolution for B={batch_idx}, Node={node_idx}')
            plt.xlabel('Feature Dimension')
            plt.ylabel('Feature Value')
            plt.legend()
            plt.show()
        # 最终投影回一维输出
        return  x.unsqueeze(-1).transpose(1, 2)

class BatchedSparseNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_mlp, aggr='add'):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.edge_mlp      = edge_mlp
        self.aggr          = aggr

    def forward(self, x, edge_index, edge_attr):
        """
        x:         [B, N, F_in]
        edge_index:[2, E]
        edge_attr: [E, 1]
        return:    [B, N_target, F_out]
        """
        B, N, Fin = x.size()
        E = edge_index.size(1)
        Fout = self.out_channels

        # 1. Edge MLP -> [E, Fin*Fout]
        W = self.edge_mlp(edge_attr)
        W = W.view(E, Fin, Fout)

        # 2. Gather source
        src = edge_index[0]
        Xs  = x[:, src, :]  # [B, E, Fin]

        # 3. Message matmul
        M = torch.einsum('bef, efo -> beo', Xs, W)  # [B, E, Fout]

        # 4. Scatter
        tgt = edge_index[1]
        N_target = N if N==x.size(1) else edge_attr.size(0)  # 仅为模板
        out = scatter(M, tgt.unsqueeze(0).expand(B, -1), dim=1, dim_size=N_target, reduce=self.aggr)
        return out


class BatchedSparseFiLMConv(nn.Module):
    def __init__(self, hidden,out_channels, aggr='add'):
        super().__init__()
        self.in_channels = hidden
        self.edge_emd = 8
        self.edge_proj = nn.Linear(1, self.edge_emd )
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.edge_emd  + 2*self.in_channels, 64),  # 多了一个 x_tgt 大小
            nn.ReLU(),
            nn.Linear(64, 2 * self.in_channels)
        )
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        B, N, Fin = x.size()
        src, tgt = edge_index
        E = src.size(0)

        x_src = x[:, src, :]     # [B, E, Fin]
        x_tgt = x[:, tgt, :]     # [B, E, Fin]
        e_proj = self.edge_proj(edge_attr)  # [E, Fin]
        e_proj = e_proj.unsqueeze(0).expand(B, -1, -1)

        # 组合边标量 + 源节点 + 目标节点
        cond = torch.cat([e_proj, x_src, x_tgt], dim=-1)    # [B, E, 1+2*Fin]
        phi = self.cond_mlp(cond)                     # [B, E, 2*Fin]
        gamma, beta = phi.chunk(2, dim=-1)            # 各 [B, E, Fin]

        # 用 FiLM 调制源节点特征
        m = gamma * x_src + beta                      # [B, E, Fin]
        out = scatter(m, tgt.unsqueeze(0).expand(B, -1),
                      dim=1, dim_size=N, reduce=self.aggr)
        return out



