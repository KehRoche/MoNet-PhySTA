import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from src.base.model import BaseModel



class MoNet(BaseModel):
    def __init__(self, input_dim,output_dim,model_config):
        super(MoNet, self).__init__(input_dim,output_dim)
        self.input_dim = output_dim

        self.A = model_config['adj']

        seq_len = model_config['seq_len']
        emd_dim = model_config['emd_dim']
        hidden_dim = model_config['hidden_dim']
        tcn_layers = model_config['tcn_layers']
        kno_layers = model_config['kno_layers']
        #model_config['covariate_dim']

        # deterministic path
        self.TempModule = TempEvo(model_config,self.input_dim,seq_len,hidden_dim, tcn_layers,kno_layers)

        self.SptialModule= SpitalDif(model_config,self.input_dim,seq_len,hidden_dim)

        self.router_weight = nn.Parameter(torch.zeros(1, 1,seq_len,emd_dim), requires_grad=True)
        self.out_fc_1   = nn.Linear(emd_dim, emd_dim//2)
        self.out_fc_2   = nn.Linear(emd_dim//2, self.input_dim)

    def forward(self, input,  label=None):
        #batch,len,nodes,feat
        input.transpose_(1, 2)
        X = input[:,:,:,:self.input_dim]
        time = input[:,:,:,self.input_dim:]
        X_inevo = self.TempModule(X,time)
        X_exdif = self.SptialModule(X,self.A,time)

        weight_inevo = 0.5*torch.ones_like(X_inevo)+self.router_weight
        weight_exdif = 0.5*torch.ones_like(X_exdif)-self.router_weight
        #+ weight_exdif*X_exdif
        y_hat = weight_inevo*X_inevo + weight_exdif*X_exdif

        #y_hat = self.test(history_data)

        y_hat =  F.relu(self.out_fc_1(y_hat))
        y_hat = self.out_fc_2(y_hat)

        # forecast    = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        # forecast    = forecast.transpose(1,2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        return  y_hat.transpose_(1, 2)



class SpitalDif(nn.Module):
    def __init__(self, config,input_dim,seq_len,hidden_dim):
        super().__init__()
        #attribute

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.emd_dim = config['emd_dim']
        self.hidden_dim = hidden_dim

        #转移矩阵
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.core_index = config['core_index']
        self._time_emb_dim = config['time_emb_dim']

        init_decay = 1
        init_lambda = 0.01
        self.num_matric = config['num_gconv']

        # start embedding layer
        self.embedding   = nn.Linear(self.input_dim, self.emd_dim)
        # time embedding
        self.T_i_D_emb  = nn.Parameter(torch.empty(288, self._time_emb_dim))
        self.D_i_W_emb  = nn.Parameter(torch.empty(7, self._time_emb_dim))

        # pivot_feature
        self.env_fea = nn.Parameter(torch.rand(1, self.emd_dim))
        self.pivot_emb = nn.Parameter(torch.empty(self.emd_dim, self.emd_dim))


        init_gamma = torch.exp(torch.tensor(1.0))
        self.gamma = nn.Parameter(init_gamma)  # 初始值为 e 的可学习参数

        self.lambda_ = nn.Parameter(torch.tensor(init_lambda))  # 初始值为 1 的可学习参数

        self.dropout = nn.Dropout(config['dropout'])
        self.gcn_updt = nn.Linear(
            self.emd_dim*self.num_matric, self.emd_dim)
        self.out_linear = nn.Linear(self.emd_dim, self.emd_dim)
        self.decay_factor = nn.Parameter(torch.tensor(init_decay, dtype=torch.float32))


        # local sim comput param
        self.gatsim = nn.Sequential(
            nn.Linear(4 * self.input_dim, self.input_dim, bias=False),  # Input is the concatenation of X_i and X_j
            nn.LeakyReLU(),
            nn.Linear(self.input_dim, 1)
        )
        self.compress = nn.Linear(self.emd_dim, 1)

        self.lambda_S = nn.Parameter(torch.tensor(init_lambda))

        #dif function param
        self.dispers =  nn.Parameter(torch.log(torch.tensor(init_decay)))  # 可学习 log 值
        self.Wd_in = nn.Parameter(torch.randn(self.emd_dim, self.emd_dim))  # 输入权重矩阵
        self.Wd_out = nn.Parameter(torch.randn(self.emd_dim, self.emd_dim))  # 输出权重矩阵

        #sipon param
        self.beta_s = nn.Parameter(torch.tensor(1, dtype=torch.float32))  # 可学习的参数 beta_s
        self.G = nn.Parameter(torch.tensor(10, dtype=torch.float32))  # 可学习的参数 G
        self.Ws_in = nn.Parameter(torch.randn(self.emd_dim, self.emd_dim))  # 输出权重矩阵

        self.init_parm()
    def init_parm(self):
        """集中式权重初始化函数"""
        #######################################
        # 1. 时间特征嵌入初始化
        #######################################
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

        #######################################
        # 2. 图卷积相关参数
        #######################################
        # GCN更新层
        nn.init.xavier_uniform_(self.gcn_updt.weight)
        nn.init.zeros_(self.gcn_updt.bias)

        # 相似度计算模块
        for layer in self.gatsim:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # 特征压缩层
        nn.init.xavier_uniform_(self.compress.weight)
        nn.init.zeros_(self.compress.bias)

        #######################################
        # 3. 环境与枢纽特征
        #######################################
        nn.init.xavier_uniform_(self.env_fea)
        nn.init.xavier_uniform_(self.pivot_emb)

        #######################################
        # 4. 扩散和虹吸参数
        #######################################
        # 扩散权重矩阵
        nn.init.xavier_uniform_(self.Wd_in)
        nn.init.xavier_uniform_(self.Wd_out)
        nn.init.xavier_uniform_(self.Ws_in)

        # 数值稳定的扩散系数
        nn.init.normal_(self.dispers, mean=0.0, std=0.1)

        # 虹吸参数
        nn.init.constant_(self.beta_s, 1.0)
        nn.init.uniform_(self.G, 0.5, 2.0)

        #######################################
        # 5. 可学习标量参数
        #######################################
        # 异常检测系数
        nn.init.uniform_(self.lambda_S, 0.5, 1.5)

        # 时序衰减因子
        nn.init.uniform_(self.decay_factor, 0.01, 0.3)

        # 扩散强度系数
        nn.init.normal_(self.gamma, mean=0.0, std=0.1)

        # 全局平衡系数
        nn.init.uniform_(self.lambda_, 0.5, 1.5)

    def gconv(self, support, X_0):
        out = [X_0]
        for graph in support:
            if len(graph.shape) == 3:  # staitic or predefined grap
                graph = graph.unsqueeze(1).repeat(1, self.seq_len, 1,1)
            H_k = torch.matmul(graph, X_0)
            out.append(H_k)
        out = torch.cat(out, dim=-1)
        out = self.gcn_updt(out)
        out = self.dropout(out)
        return out

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

        # adj_indices = torch.nonzero(A)  # 获取所有存在边的 (i, j) 索引
        # num_edges = adj_indices.shape[0]
        # local_sim = torch.zeros(X.shape[0], X.shape[1], N,1,
        #                     device=X.device)  # (batch, seq_len, nodes, 2*feat)
        #
        # A_inv = torch.sigmoid(torch.where(A != 0, 1 / A, torch.zeros_like(A)))
        # for k in range(num_edges):
        #     batch_idx, i, j = adj_indices[k]  # 获取 (batch, i, j) 形式的索引
        #     X_i = X[batch_idx, :, i, :]  # 取 X_i
        #     X_j = X[batch_idx, :, j, :]  # 取 X_j
        #     ij_sim = self.gatsim(torch.cat([X_i, X_j], dim=-1)).squeeze(-1)
        #     local_sim[batch_idx, :, i, 0] += ij_sim*A_inv[batch_idx,i,j] # (batch, seq_len, 1, 1)
        #X_cat = X_cat.contiguous().view(X.size(0), X.size(1), N, N, 2 * self.input_dim)
        #fea_sum = torch.einsum("bnn,bsnf->bsnf", A, self.compress(X))


        # 假设 X 和 A 已经定义，且 X 形状为 (batch_size, seq_len, N, input_dim)
        N = X.size(2)
        #A_inv = torch.sigmoid(torch.where(A != 0, 1 / A, torch.zeros_like(A)))  # (batch_size, N, N)

        # 1. 扩展 X，准备进行节点间的相似度计算

        # 2. 计算节点对之间的相似度矩阵
        # 使用自定义的相似度计算函数 gatsim 计算节点对的相似度
        # 我们将 X_i 和 X_j 的拼接矩阵计算为一个大矩阵，然后进行批量计算

        sim_matrix = torch.matmul(X, X.transpose(-1, -2))  # (batch_size * seq_len, N, N, 1)

        # 3. 进行缩放，防止数值过大或过小
        # 假设 input_dim 为 X 的最后一个维度（特征维度），通常为 Q 和 K 的维度
        d_k = X.size(-1)  # 即 input_dim
        sim_matrix = sim_matrix / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # 去掉最后一个维度，得到节点对之间的相似度矩阵
        sim_matrix = sim_matrix.squeeze(-1)  # (batch_size * seq_len, N, N)

        # 4. 计算加权相似度
        sim_matrix = sim_matrix * A # (batch_size, seq_len, N, N))
        sim_matrix = torch.sigmoid(torch.matmul(sim_matrix, torch.ones(size=[N,1]).to(X.device)))
        # 5. 计算特征加权和
        fea_sum = torch.einsum("nn,bsnf->bsnf", A, self.compress(X)) # (batch_size, seq_len, N, input_dim)

        # 6. 计算异常度

        # self.lambda_L = torch.sqrt(torch.tensor(N, dtype=torch.float32)) * self.lambda_S
        # anomalyL = torch.max(torch.zeros_like(sim_matrix), (
        #             sim_matrix * (1.0 / fea_sum)) - self.lambda_L)  # Shape (batch_size, seq_len, N, N)
        # anomalyS = torch.max(torch.zeros_like(sim_matrix),
        #                      sim_matrix - self.lambda_S)  # Shape (batch_size, seq_len, N, N)
        #
        # # 最终的异常度：L 和 S 异常度的合成
        # anomaly_degree = anomalyL + anomalyS  # Shape (batch_size, seq_len, N, N)

        K = self.gatsim(torch.cat([fea_sum, sim_matrix,Time.transpose(1,2)], dim=-1))
        return K

    def Disp(self,A,K):
        # Step 1: 扩展 alpha 为 (N, N)
        N = K.shape[2]
        #alpha_expanded = K.unsqueeze(-1)  # (N, 1)
        A = A.reshape(1,1,N,N).repeat(K.shape[0], K.shape[1],1,1)
        # Step 2: 计算扩散矩阵
        #F_dif =  (A ** (K)) * torch.exp(-self.dispers *A)  # (N, N)

        # A_log_K = torch.where(A != 0, torch.log(A) * K.unsqueeze(-1), torch.zeros_like(A))  # (b, l, n, n)
        # exp_term = torch.exp(-self.dispers * A)
        # F_dif = A_log_K * exp_term  # 这里通过分步计算，确保数值稳定

        F_dif = A * (K+torch.ones_like(K))
        # Step 3: 避免数值问题
        F_dif = torch.where(A > 0, F_dif, torch.zeros_like(F_dif))  # 排除非连接的节点

        return F_dif

    def Sipon(self,A):
        D = A.sum(dim=-1)
        F = self.G * (D.unsqueeze(-1) * D.unsqueeze(1)) / (A + 1e-6)
        # 计算虹吸效应强度 a_ij^i 和 a_ij^j
        a_i = F / D.unsqueeze(-1)  # (B, N, N)
        a_j = F / D.unsqueeze(1)   # (B, N, N)

        siphon_coeff = torch.tanh(self.beta_s * ((a_i / (a_i + a_j + 1e-6)) - 0.5))
        A_sip = A * siphon_coeff  # 原始邻接矩阵与虹吸系数元素乘积，(B, N, N)

        return A_sip



    def CoreNodeEnhancer(self, X):
        """
        :param X: Input feature matrix (B, N, D)
        :return: Enhanced feature matrix (B, N, D)
        """
        B, L, N, D = X.shape
        assert D == self.emd_dim, "Feature dimension mismatch"

        # Create a mask for core nodes
        mask = torch.zeros(N, device=X.device)
        mask[self.core_index] = 1  # Set core nodes to 1
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)

        # Calculate enhancement
        enhanced_features = torch.matmul(X, self.pivot_emb)  # (B, N, D)
        enhanced_X = X + mask * (enhanced_features - X)  # Apply only to core nodes
        return enhanced_X
    def forward(self, X_sptial,A,Time):
        #batch,nodes,seq,feat
        X = self.embedding(X_sptial).transpose(1, 2)

        #b,L,N,F
        # X_env =self.env_fea.repeat(X.shape[0],X.shape[1],X.shape[2],1)
        # decay_weights = torch.exp(-self.decay_factor * torch.arange(self.seq_len).view(1, -1, 1, 1).to(self.env_fea.device))
        # decayed_env_fea =X_env * decay_weights
        # #环境特征衰减
        # X = X + decayed_env_fea
        #锚点特征增强
        #X = self.CoreNodeEnhancer(X)
        K = self.anomaly_factors(X, A,Time)
        A_dif = self.Disp(A,K)

        A_sip = self.Sipon(A)


        # D = A.sum(dim=-1)
        # D_inv = D.unsqueeze(-1).pow(-1)  # (B, N, 1)，每个节点的度的倒数
        # #计算 (D^{-1} * A_{dif}^t) * X_t * W_{in}^{dif}
        # term1 = D_inv * torch.matmul(A_dif.transpose(-1, -2), X)  # (B, N, N) * (B, N, F) -> (B, N, F)
        # term1 = torch.matmul(term1, self.Wd_in)
        #
        # # 计算 (D^{-1} * A_{dif}^t) * X_t * W_{out}^{dif}
        # term2 = D_inv * torch.matmul(A_dif, X)  # (B, N, N) * (B, N, F) -> (B, N, F)
        # term2 = torch.matmul(term2, self.Wd_out)  # (B, N, F) * (F, F') -> (B, N, F')
        #
        # # 计算最终输出
        # X_dif = torch.relu(term1 - term2)  # 进行 ReLU 激活
        #
        # X_sip = D_inv * torch.matmul(A_sip.transpose(-1, -2), X_dif)  # (B, N, N) * (B, N, F) -> (B, N, F)
        # X_sip = torch.relu(torch.matmul(X_sip, self.Wd_in))

        support = []
        support.append(A_dif)
        #support.append(A_dif.transpose(-1,-2))
        support.append(A_sip.transpose(-1,-2))
        support.extend(self._multi_order(A,order=2))
        #扩散衰减矩阵聚合+原始图卷积
        Y_dif = self.gconv(support,X)
        Y_dif = self.dropout(Y_dif)
        Y_dif = self.out_linear(Y_dif)
        return Y_dif.transpose(1,2)



class SubSeqForcast(nn.Module):
    def __init__(self, config,seq_len,kno_layers=4, linear_type=True, normalization=False):
        super(SubSeqForcast, self).__init__()
        self.op_size = config['temp_ampify']*seq_len
        self.freq_modes = config['freq_modes']
        self.layers = kno_layers
        self.linear_type = linear_type
        self.normalization = normalization
        #capture high freq info
        self.highfreq_compen = Conv1d(self.op_size, self.op_size, 1)
        # Encoder and Decoder
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, self.op_size),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.op_size, seq_len)
        )

        # Koopman Operator Layer
        self.koopman_layer = KoopmanOperator1D(self.op_size, modes_x=self.freq_modes)

        # Optional Normalization Layer
        if self.normalization:
            self.norm_layer = nn.BatchNorm1d(self.op_size)

    def forward(self, x):
        # Encoding
        x = x.transpose(-1, -2)
        x_encoded = self.encoder(x)

        # Koopman Operator Dynamics
        x_dynamic = x_encoded
        for _ in range(self.layers):
            x_dynamic = self.koopman_layer(x_dynamic)
            if not self.linear_type:
                x_dynamic = torch.tanh(x_dynamic)

        # Optional Normalization
        x_dynamic = x_dynamic.transpose(-1, -2)
        if self.normalization:
            x_dynamic = self.norm_layer(self.highfreq_compen(x_dynamic).transpose(-1, -2)+x_dynamic)
        else:
            x_dynamic = self.highfreq_compen(x_dynamic).transpose(-1, -2)+x_encoded
        # Decoding
        x_reconstructed = self.decoder(x_dynamic)
        return x_reconstructed.transpose(-1, -2)


class LongtermForcast(nn.Module):
    def __init__(self,config, seq_len,input_hidden, output_hidden, tcn_layers, revin=False):
        super().__init__()

        self.input_hidden = input_hidden
        self.output_hidden = output_hidden
        self.seq_len = seq_len

        self.num_nodes = config['num_nodes']

        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.kernel_size = config['kernel_size']
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.dropout = config['dropout']
        self.head_dropout = config['head_dropout']
        self.depth = tcn_layers

        self.DSC_blocks = nn.ModuleList([DSCLayer(input_dim=self.input_hidden, out_dim=self.output_hidden, kernel_size=self.kernel_size) for _ in range(self.depth)])
        self.W_P = nn.Linear(self.output_hidden, self.output_hidden)
        self.head0 = nn.Sequential(
            nn.Linear(self.input_hidden, self.output_hidden),
            nn.GELU(),
            nn.Dropout(self.head_dropout),

        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.seq_len * self.input_hidden, self.seq_len *self.input_hidden ),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.seq_len *self.input_hidden, self.output_hidden * self.seq_len),
            nn.Dropout(self.head_dropout)
        )
        self.dropout = nn.Dropout(self.dropout)
        self.revin = RevIN(self._num_node) if revin else None

    def forward(self, x_emd):
        bs, num_node, _, _ = x_emd.shape
        if self.revin:
            x_emd = self.revin(x_emd, 'norm')
        x_emd = self.dropout(x_emd)
        u = self.head0(x_emd)
        x_emd = x_emd.transpose(-1, -2)
        for DSC_block in self.DSC_blocks:
            x_emd = DSC_block(x_emd)
        x_emd = x_emd.transpose(-1, -2)
        x_emd = self.head1(x_emd)
        x_emd = x_emd.reshape(bs,num_node, self.seq_len, -1)
        x = u + x_emd
        out = self.W_P(x)
        if self.revin:
            out = self.revin(out, 'denorm')
        return out

class TempEvo(nn.Module):
    def __init__(self,config,input_dim,seq_len,hidden_dim,kno_layers,tcn_layers):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        self.dropout = config['dropout']
        self.device= config['device']
        self.emd_dim = config['emd_dim']
        self.hidden_channels = config['hidden_channels']
        side_channels = [config['covariate_dim']]+ self.hidden_channels
        self.hidden_channels = [self.emd_dim] +  self.hidden_channels

        #KNO parm
        self.kno_layers = kno_layers
        self.tcn_layers = tcn_layers

        self.feature_embedding = Conv1d(input_dim, self.emd_dim, 1, actv=False)
        self.side_encoding = nn.ModuleList([Conv1d(side_channels[i], side_channels[i+1], 1, dropout=self.dropout) for i in range(len(side_channels) - 1)])

        self.route_weight = nn.Parameter(torch.randn(self.seq_len, self.seq_len))

        self.PINN = SubSeqForcast(config,seq_len=self.seq_len,kno_layers=self.kno_layers)
        self.DNN = nn.ModuleList([LongtermForcast(config,seq_len=self.seq_len,
                                                 input_hidden=self.hidden_channels[i], output_hidden=self.hidden_channels[i+1],tcn_layers=self.tcn_layers) for i in range(len(self.hidden_channels) - 1)])
        self.route_MLP = Residual(MLP(self.emd_dim,hidden_dim=self.emd_dim))

        self.residual = nn.ModuleList([Conv1d(self.hidden_channels[i], self.hidden_channels[i+1], 1, actv=False) for i in range(len(self.hidden_channels)-1)])
        self.dnn_output = nn.Linear(self.hidden_channels[-1],self.emd_dim)
        #Opt Setting
        self.loss = torch.nn.MSELoss()
        self.router_weight = nn.Parameter(torch.zeros(1, 1,self.seq_len,self.emd_dim), requires_grad=True)


    def forward(self, x,x_time):
        batch_size = x.size(0)
        l_recons = 0
        #batch,nodes,len,feat
        x = self.feature_embedding(x.transpose(-1, -2))
        x = x.transpose(-1, -2)
        y_pinn = self.PINN(x)

        #x：batch,num_node,seq_len,feat
        for i in range(len(self.hidden_channels) - 1):
            x_resi = x.clone()
            y_dnn = self.DNN[i](x)
            x_time = self.side_encoding[i](x_time.transpose(-1, -2)).transpose(-1, -2)
            x = torch.relu(y_dnn+x_time+self.residual[i]((x_resi.transpose(-1, -2))).transpose(-1, -2))
        y_dnn = self.dnn_output(x)
        #l_recons += self.loss(x_re.reshape(batch_size,-1),x.reshape(batch_size,-1))
        #l_pred = self.loss(y_pred.reshape(batch_size,-1),y.reshape(batch_size,-1))


        weight_AI = 0.5*torch.ones_like(y_dnn)+self.router_weight
        weight_Physics = 0.5*torch.ones_like(y_dnn)-self.router_weight
        y_t = weight_AI*y_dnn + weight_Physics*y_pinn
        y_t = self.route_MLP(y_t)
        #loss = 5 * l_pred + 0.5 * l_recons
        return y_t

class KoopmanOperator1D(nn.Module):
    def __init__(self, op_size, modes_x=16):
        super(KoopmanOperator1D, self).__init__()
        self.op_size = op_size
        self.modes_x = modes_x
        self.scale = 1 / (op_size * op_size)
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(op_size, op_size, self.modes_x, dtype=torch.cfloat))

    def time_marching(self, input, weights):
        return torch.einsum("bntx,tfx->bnfx", input, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        # Fourier Transform
        x_ft = torch.fft.rfft(x.transpose(-1, -2))
        # Koopman Operator Time Marching
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes_x] = self.time_marching(x_ft[:, :, :, :self.modes_x], self.koopman_matrix)
        # Inverse Fourier Transform
        x_out = torch.fft.irfft(out_ft, n=x.size(-2))
        return x_out.transpose(-1, -2)

class DSCLayer(nn.Module):
    def __init__(self, input_dim, out_dim, kernel_size=4):
        super().__init__()
        self.Resnet = nn.Sequential(
            Conv1d(input_dim, input_dim, kernel_size=kernel_size, groups=input_dim, padding='same'),
            nn.GELU()
        )
        self.Conv_1x1 = nn.Sequential(
            Conv1d(input_dim, input_dim, kernel_size=1),
            nn.GELU()
        )
        self.bn = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        x = self.Conv_1x1(x + self.Resnet(x))
        b, n = x.shape[:2]
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = x.reshape([b, n, -1, x.shape[-1]])
        return x

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,groups=1, kernel_size=1,padding='same', dropout=0.1, actv=True):
        super(Conv1d, self).__init__()
        self.convolution = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.actv = actv

    def forward(self, x):
        if len(x.shape) == 4:
            b, n = x.shape[:2]
            x = x.flatten(0, 1)
            y = self.convolution(x)
            y = y.reshape([b, n, -1, y.shape[-1]])
        else:
            y = self.convolution(x)
        if self.actv:
            y =  torch.relu(self.dropout(y))
        return y