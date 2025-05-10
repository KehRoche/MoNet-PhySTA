import torch
import torch.nn as nn
from collections import defaultdict

from sympy import false

from .UniMoudle import *
from .TempEvo import SubSeqForcast

from torch_geometric.nn import NNConv
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter


# class MSKGN(torch.nn.Module):
#     def __init__(self,hidden_channels, levels, kernel_nn_hidden):
#         super().__init__()
#         self.levels = levels
#         # Build per-level convs
#         self.convs = torch.nn.ModuleList()
#         for l in range(levels):
#             kernel_nn = torch.nn.Sequential(
#                 torch.nn.Linear(1, kernel_nn_hidden),
#                 torch.nn.ReLU(),
#                 torch.nn.Linear(kernel_nn_hidden, hidden_channels * hidden_channels)
#             )
#             self.convs.append(BatchedSparseNNConv(hidden_channels, hidden_channels, kernel_nn, aggr='mean'))
#         # Output projection
#         self.fc_out = torch.nn.Linear(hidden_channels, 1)
#
#     def multiscale_graph(self,adj_matrix: torch.Tensor,
#                                               node_feats: torch.Tensor,
#                                               levels: int,
#                                               gamma: float = 0.5,
#                                               topk: int = 5):
#         """
#         构造多尺度图，仅返回每层的节点映射、边索引、边权重。
#         下采样策略：Louvain 社区检测 -> 度中心诱导点 -> Nyström 近似 + Top-k 稀疏化
#
#         Args:
#             adj_matrix: [N,N] 原始邻接矩阵
#             node_feats: [N,d]  每个节点的特征，用于计算 RBF 核
#             levels:     层数 L（最细到最粗）
#             gamma:      RBF 核参数
#             topk:       粗图每行保留 Top-k 边
#
#         Returns:
#             clusters:      List[LongTensor] 长度 L，每项 [n_l] 表示第 l 层的“节点索引”
#             edge_indices: List[LongTensor] 长度 L，每项 [2, E_l] 的边列表
#             edge_weights: List[Tensor]       长度 L，每项 [E_l] 的边权
#         """
#         device = adj_matrix.device
#         N = adj_matrix.size(0)
#
#         # 初始：第 0 层就是原图
#         cluster = torch.arange(N, device=device)  # 每个节点映射到自己
#         edge_index, edge_weight = dense_to_sparse(adj_matrix)
#
#         clusters, edge_indices, edge_weights = [], [], []
#
#         # RBF 核函数
#         def rbf_kernel(X1, X2):
#             diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # n×m×d
#             D2 = (diff ** 2).sum(dim=2)  # n×m
#             return torch.exp(-gamma * D2)  # n×m
#
#         for l in range(levels):
#             clusters.append(cluster.clone())
#             edge_indices.append(edge_index.clone())
#             edge_weights.append(edge_weight.clone())
#
#             if l == levels - 1:
#                 break
#
#             # ——— 构造下一层 超图 A_coarse ———
#             # 1) 从当前层 edge_index + edge_weight 构造 NetworkX 图
#             G = nx.Graph()
#             ei = edge_index.cpu().numpy()
#             ew = edge_weight.cpu().numpy()
#             for idx in range(ei.shape[1]):
#                 u, v = ei[0, idx], ei[1, idx]
#                 G.add_edge(int(u), int(v), weight=float(ew[idx]))
#
#             # 2) Louvain 社区检测
#             part = community_louvain.best_partition(G, weight='weight')
#             communities = {}
#             for node, com in part.items():
#                 communities.setdefault(com, []).append(node)
#
#             # 3) 每个社区选度最大的节点作为诱导点
#             centers = []
#             deg = dict(G.degree(weight='weight'))
#             for nodes in communities.values():
#                 centers.append(max(nodes, key=lambda i: deg[i]))
#             m = len(centers)
#
#             # 4) 计算 Nyström 近似的 A_coarse
#             X = node_feats
#             Xc = X[centers]  # m×d
#             K_mm = rbf_kernel(Xc, Xc)  # m×m
#             K_nm = rbf_kernel(X, Xc)  # n×m
#             W = K_nm @ torch.linalg.pinv(K_mm)  # n×m
#             A_coarse = (W.t() @ K_nm).to(device)  # m×m
#             A_coarse.fill_diagonal_(0)
#
#             # 5) Top-k 稀疏化
#             for i in range(m):
#                 row = A_coarse[i]
#                 idx = torch.topk(row, topk, largest=True).indices
#                 mask = torch.zeros_like(row)
#                 mask[idx] = 1
#                 A_coarse[i] = row * mask
#
#             # 6) 构造下一层的 cluster 映射
#             # new_cluster 长度 n_l，值为 0..m-1（每个原节点映射到哪个中心）
#             # 这里用 argmax(W[i]) 做最近中心分配，也可用 communities
#             assignment = torch.argmax(K_nm, dim=1)  # n
#             cluster = assignment  # 下一层映射
#
#             # 7) 更新 edge_index, edge_weight 到 coarse
#             edge_index, edge_weight = dense_to_sparse(A_coarse)
#
#         return clusters, edge_indices, edge_weights
#
#     def forward(self, x, adj_matrix):
#         device = x.device
#         x = x.squeeze(-1).transpose(-1, -2)
#         pools, graphs = [], []
#         cluster, edge_index, edge_weight = self.multiscale_graph(adj_matrix, x, l)
#         #downward
#         for l in range(range(self.levels - 1)):
#             local = cluster[l]
#             x = x[local]
#             local_index, local_weight = edge_index[l], edge_weight[l]
#             x = x+(self.convs[l](x, local_index, local_weight))
#             x = F.relu(x)
#         ##upward
#         for l in reversed(range(self.levels - 1)):
#             local = cluster[l]
#             x = x[local]
#             local_index, local_weight = edge_index[l], edge_weight[l]
#             x = x+(self.convs[l](x, local_index, local_weight))
#             x = F.relu(x)
#             up_index,up_weight = edge_index[l+1], edge_weight[l+1]
#             up = cluster[l+1]
#             for u, v in local_index:
#                 mem_u = [n for n, p in part.items() if p == part[u]]
#                 mem_v = [n for n, p in part.items() if p == part[v]]
#                 up_index += [(i, j) for i in mem_u for j in mem_v]
#             x = x[up]
#             x = x+(self.convs[l](x, up_index, up_weight))
#             x = F.relu(x)
#
#         return self.fc_out(x)
#
# class BatchedSparseNNConv(nn.Module):
#     def __init__(self, in_channels, out_channels, edge_mlp, aggr: str = 'add'):
#         """
#         Args:
#             in_channels: F_in
#             out_channels: F_out
#             edge_mlp: nn.Module mapping [E, D] -> [E, F_in * F_out]
#             aggr: 'add' | 'mean' | 'max'
#         """
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.edge_mlp = edge_mlp
#         self.aggr = aggr
#
#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
#                 edge_attr: torch.Tensor) -> torch.Tensor:
#         """
#         x:         [B, N, F_in]
#         edge_index:[2, E]
#         edge_attr: [E, D]
#         return:    [B, N, F_out]
#         """
#         B, N, F_in = x.size()
#         _, E = edge_attr.size(0), edge_index.size(1)
#         F_out = self.out_channels
#
#         # 1. 边属性映射 → W: [E, F_in, F_out]
#         W = self.edge_mlp(edge_attr)                               # [E, F_in*F_out]
#         W = W.view(E, F_in, F_out)                                 # [E, F_in, F_out]
#
#         # 2. 批量提取源节点特征 X_src: [B, E, F_in]
#         src_idx = edge_index[0]                                    # [E]
#         X_src = x[:, src_idx, :]                                   # [B, E, F_in]
#
#         # 3. 批量矩阵乘法 → M: [B, E, F_out]
#         # 相当于 for each e: M[:,e,:] = X_src[:,e,:] @ W[e]
#         M = torch.einsum('bef, efo -> beo', X_src, W)              # [B, E, F_out] :contentReference[oaicite:3]{index=3}
#
#         # 4. 根据 target_idx 聚合消息 → out: [B, N, F_out]
#         tgt_idx = edge_index[1]                                    # [E]
#         out = scatter(M, tgt_idx.unsqueeze(0).expand(B, -1),
#                       dim=1, dim_size=N, reduce=self.aggr)        # [B, N, F_out] :contentReference[oaicite:4]{index=4}
#
#         return out
#

import torch
import torch.nn.functional as F
from torch import nn
import networkx as nx
import community as community_louvain
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter

class MSKGN(nn.Module):
    def __init__(self, hidden_channels: int, levels: int, adj_matrix,kernel_nn_hidden: int, topk: int = 5, gamma: float = 0.5):
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
        self.gamma = gamma
        self.vis = false
        # down / mid / up 三组卷积列表
        self.conv_down = nn.ModuleList()
        self.conv_mid  = nn.ModuleList()
        self.conv_up   = nn.ModuleList()
        for l in range(levels):
            # 同层 mid-scale 卷积
            mlp_mid = nn.Sequential(
                nn.Linear(1, kernel_nn_hidden),
                nn.ReLU(),
                nn.Linear(kernel_nn_hidden, hidden_channels*hidden_channels)
            )
            self.conv_mid.append(BatchedSparseNNConv(hidden_channels, hidden_channels, mlp_mid, aggr='mean'))


            # 下行跨层 conv_down[l]: l->l+1
            mlp_down = nn.Sequential(
                nn.Linear(1, kernel_nn_hidden),
                nn.ReLU(),
                nn.Linear(kernel_nn_hidden, hidden_channels*hidden_channels)
            )
            self.conv_down.append(BatchedSparseNNConv(hidden_channels, hidden_channels, mlp_down, aggr='mean'))

            # 上行跨层 conv_up[l]: (l+1)->l
            mlp_up = nn.Sequential(
                nn.Linear(1, kernel_nn_hidden),
                nn.ReLU(),
                nn.Linear(kernel_nn_hidden, hidden_channels*hidden_channels)
            )
            self.conv_up.append(BatchedSparseNNConv(hidden_channels, hidden_channels, mlp_up, aggr='mean'))
        self.center,self.part,self.mg = self.multiscale_graph(adj_matrix.squeeze(-1))
        self.projs = nn.ModuleList([
            nn.Linear( 3 * hidden_channels, hidden_channels),
            nn.ReLU()]
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

        # 递归粗化
        for l in range(self.levels):
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
            A.fill_diagonal_(0)
            A_up = A.clone()
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
        for t in range(1):  # 如果需要多次迭代，可改为 range(depth)
            # DOWNWARD: l=0..L-2
            for l in range(self.levels):
                ei, ew = mg['edge_down'][l]
                x_l = x.clone()  # 当前层 feature
                # 聚合到粗层节点 order 与 clusters[l+1] 对齐
                msg = self.conv_down[l](x_l, ei, ew)  # [B, n_{l+1}, F]
                # 将 msg 对应位置取出
                for node, comm in self.part.items():
                    center = self.center[comm]  # 找到当前节点所属社区的中心节点
                    x_l[:,node,:] = msg[:,center,:]  # 将中心节点特征映射给该节点
                features[f'down_l{l}'] = x_l
                # 同层 mid
                ei_m, ew_m = mg['edge_mid'][l]
                x_m = x + self.conv_mid[l](x, ei_m, ew_m)
                #x_m = F.relu(x_m)
                features[f'mid_l{l}'] = x_m

                # # up-scale
                # ei_u, ew_u = mg['edge_up'][l]
                # msg_up = x+self.conv_up[l](x, ei_u, ew_u)
                # msg_up = F.relu(msg_up)
                # features[f'up_l{l}'] = msg_up

                concat = torch.cat([x, x_l, x_m], dim=-1)  # [B,N,F+3H]
                fused = F.relu(self.projs[l](concat))
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
        return fused.unsqueeze(-1).transpose(1, 2)

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



