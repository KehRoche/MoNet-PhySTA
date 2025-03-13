import torch
import torch.nn as nn
from tslearn.metrics import cdist_dtw
from .UniMoudle import *
from .TempEvo import SubSeqForcast



class SpitalDif(nn.Module):
    def __init__(self, config,input_dim,seq_len,hidden_dim):
        super().__init__()
        #attribute

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.emd_dim = config['emd_dim']
        hidden_dims = config['hidden_channels']
        #转移矩阵
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.core_index = config['core_index']
        self._time_emb_dim = config['time_emb_dim']

        init_decay = 1
        init_lambda = 0.01
        num_matric = config['num_gconv']

        #self.gcn = UnetGCN(hidden_dims,num_matric)
        self.gcn = GraphConvLayer(self.emd_dim,self.emd_dim,num_matric)
        self.dropout = nn.Dropout(config['dropout'])

        self.out_linear = nn.Linear(self.emd_dim, self.emd_dim)
        #PINN
        #self.phy_field = SubSeqForcast(config,self.seq_len)

        # local sim comput param
        self.gatsim = nn.Sequential(
            nn.Linear(4 * self.input_dim, self.input_dim, bias=False),  # Input is the concatenation of X_i and X_j
            nn.LeakyReLU(),
            nn.Linear(self.input_dim, 1)
        )
        self.compress = nn.Linear(self.emd_dim, 1)

        #sipon param
        self.beta_coff = nn.Parameter(torch.tensor(1, dtype=torch.float32))  # 可学习的参数 beta_s
        self.G_coff = nn.Parameter(torch.tensor(1, dtype=torch.float32))  # 可学习的参数 G

        #self.init_parm()

    def _multi_order(self, graph,order):
        graph_ordered = []
        k_1_order = graph               # 1 order
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, order+1):     # e.g., order = 3, k=[2, 3]; order = 2, k=[2]
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered
    def anomaly_factors(self,X, A,Time):
        N = X.size(2)
        sim_matrix = torch.matmul(X, X.transpose(-1, -2))  # (batch_size * seq_len, N, N, 1)
        d_k = X.size(-1)  # 即 input_dim
        sim_matrix = sim_matrix / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # 去掉最后一个维度，得到节点对之间的相似度矩阵
        sim_matrix = sim_matrix.squeeze(-1)  # (batch_size * seq_len, N, N)
        sim_matrix = torch.sigmoid(sim_matrix)
        # 5. 计算特征加权和
        fea_sum = torch.einsum("nn,bsnf->bsnf", A, self.compress(X)) # (batch_size, seq_len, N, input_dim)
        sem_matrix = torch.matmul(sim_matrix,fea_sum.repeat(1,1,1,N))
        return sem_matrix

    def DTWsim(self,X):
        dtw_sim = DTWSimilarity(gamma=0.3)
        sim_matrix = dtw_sim(X.transpose(1, 2))
        return sim_matrix

    def Sipon(self,A):
        D = A.sum(dim=-1)
        F = self.G_coff * (D.unsqueeze(-1) * D.unsqueeze(1)) / (A + 1e-6)
        # 计算虹吸效应强度 a_ij^i 和 a_ij^j
        a_i = F / D.unsqueeze(-1)  # (B, N, N)
        a_j = F / D.unsqueeze(1)   # (B, N, N)

        siphon_coeff = torch.tanh(self.beta_coff * ((a_i / (a_i + a_j + 1e-6)) - 0.5))
        A_sip = A * siphon_coeff  # 原始邻接矩阵与虹吸系数元素乘积，(B, N, N)

        return A_sip


    def forward(self, X_sptial,A,Time):
        #batch,nodes,seq,feat
        X_sptial = X_sptial.transpose(1, 2)
        #X_phy = self.phy_field(X_sptial)
        #b,L,N,F
        # #环境特征衰减
        # X = X + decayed_env_fea
        #锚点特征增强
        #X = self.CoreNodeEnhancer(X)
        #sem_matrix = self.anomaly_factors(X_sptial, A,Time)
        #A_sip = self.Sipon(A)
        #DTW_sim = self.DTWsim(X_sptial)
        support = []
        #support.append(sem_matrix)
        #support.append(DTW_sim.unsqueeze(1))
        #support.append(A_sip.transpose(-1,-2))
        #support.append(A)
        support.extend(self._multi_order(A,order=3))
        #扩散衰减矩阵聚合+原始图卷积
        Y_dif = self.gcn(support,X_sptial)
        return Y_dif.transpose(1,2)


class UnetGCN(nn.Module):
    def __init__(self,hidden_dim,num_adjs):
        super().__init__()
        self.up_gconv = nn.ModuleList(
            [GraphConvLayer(hidden_dim[i], hidden_dim[i+1],num_adjs) for i in range(len(hidden_dim) - 1)])
        self.down_gconv = nn.ModuleList(
            [GraphConvLayer(hidden_dim[i]*2, hidden_dim[i - 1],num_adjs) for i in range(len(hidden_dim) - 1, 0, -1)])

    def forward(self,X_sptial,support):
        X_up = [X_sptial]
        for i in range(len(self.up_gconv)):
            X_up.append(self.up_gconv[i](support, X_up[i]))
        X_down = [X_up[-1]]
        for i in range(len(self.down_gconv)):
            X_down.append(self.down_gconv[i](support,torch.cat([X_up[-i-1], X_down[i]],dim=-1)))
        return X_down[-1]

class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,num_adjs):
        super(GraphConvLayer, self).__init__()
        self.out_channels = out_channels
        self.gcn_updt = nn.Linear(in_channels*num_adjs, out_channels)
        self.fc_list_updt = nn.Linear(
            in_channels, in_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self,support,X):
        X = self.fc_list_updt(X)
        out = [X]
        for graph in support:
            H_k = torch.matmul(graph, X)
            out.append(H_k)
        out = torch.cat(out, dim=-1)
        out = self.gcn_updt(out)
        out = self.act((self.bn(out.transpose(1,3))))
        return out.transpose(1,3)
