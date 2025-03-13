import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .UniMoudle import *



class TempEvo(nn.Module):
    def __init__(self,config,input_dim,seq_len,hidden_dim,kno_layers,tcn_layers):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        n_heads = config['n_heads']
        self.dropout = config['dropout']
        self.device= config['device']
        self.emd_dim = config['emd_dim']
        self.hidden_channels = config['hidden_channels']
        side_channels = [config['covariate_dim']]+ self.hidden_channels
        self.hidden_channels = [self.emd_dim] +  self.hidden_channels

        #KNO parm
        self.kno_layers = kno_layers
        self.tcn_layers = tcn_layers

        # self.feature_embedding = Conv1d(input_dim, self.emd_dim, 1, actv=False)
        # self.side_encoding = nn.ModuleList([Conv1d(side_channels[i], side_channels[i+1], 1, dropout=self.dropout) for i in range(len(side_channels) - 1)])
        #
        # #self.PINN = SubSeqForcast(config,seq_len=self.seq_len,kno_layers=self.kno_layers)
        # self.DNN = nn.ModuleList([LongtermForcast(config,seq_len=self.seq_len,
        #                                         input_hidden=self.hidden_channels[i], output_hidden=self.hidden_channels[i+1],tcn_layers=self.tcn_layers) for i in range(len(self.hidden_channels) - 1)])
        # #self.route_MLP = Residual(MLP(self.emd_dim,hidden_dim=self.emd_dim))
        # self.AccidentEnh = MultiHeadLocalAttention(self.emd_dim,n_heads)
        # self.gate = nn.Parameter(torch.randn(1))
        #
        # self.residual = nn.ModuleList([Conv1d(self.hidden_channels[i], self.hidden_channels[i+1], 1, actv=False) for i in range(len(self.hidden_channels)-1)])
        # self.dnn_output = nn.Linear(self.hidden_channels[-1],self.emd_dim)
        #Opt Setting
        self.loss = torch.nn.MSELoss()
        #self.router_weight = nn.Parameter(torch.zeros(1, 1,self.seq_len,self.emd_dim), requires_grad=True)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.emd_dim * self.seq_len, out_channels=self.emd_dim * self.seq_len, kernel_size=(1, 1), bias=True)
        self.encoder = nn.Sequential(*[MultiLayerPerceptron(self.emd_dim * self.seq_len, self.emd_dim * self.seq_len) for _ in range(self.tcn_layers)])

    def forward(self, x,x_time):
        l_recons = 0
        #batch,nodes,len,feat
        b,n,t,f = x.shape
        #y_pinn = self.PINN(x)
        #y_acc = self.AccidentEnh(x)

        # for i in range(len(self.hidden_channels) - 1):
        #     x_resi = x.clone()
        #     y_dnn = self.DNN[i](x)
        #     x_time = self.side_encoding[i](x_time.transpose(-1, -2)).transpose(-1, -2)
        #     x = F.gelu(y_dnn+x_time+self.residual[i]((x_resi.transpose(-1, -2))).transpose(-1, -2))
        # y_dnn = self.dnn_output(x)
        x = x.view(b,n,-1).transpose(1,2).unsqueeze(-1).contiguous()
        x = self.encoder(x)
        y = self.time_series_emb_layer(x)
        y = y.squeeze(-1).transpose(1,2).view(b,n,t,f).contiguous()


        #y = torch.sigmoid(self.gate) * y_dnn + (1 - torch.sigmoid(self.gate)) * y_acc
        # weight_AI = 0.5*torch.ones_like(y_dnn)+self.router_weight
        # weight_Physics = 0.5*torch.ones_like(y_pinn)-self.router_weight
        # y_t =weight_AI*y_dnn+ weight_Physics*y_pinn
        # y_t = self.route_MLP(y_t)
        return y

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

class KoopmanOperator1D(nn.Module):
    def __init__(self, op_size, modes_x=16):
        super(KoopmanOperator1D, self).__init__()
        self.op_size = op_size
        self.modes_x = modes_x
        self.scale = 1 / (op_size * op_size)
        # 初始化实部和虚部的参数
        self.koopman_matrix_real = nn.Parameter(self.scale * torch.rand(op_size, op_size, self.modes_x))
        self.koopman_matrix_imag = nn.Parameter(self.scale * torch.rand(op_size, op_size, self.modes_x))

    def time_marching(self, input_real, input_imag, weights_real, weights_imag):
        # 分别处理实部和虚部
        real_part = torch.einsum("bntx,tfx->bnfx", input_real, weights_real) - torch.einsum("bntx,tfx->bnfx", input_imag, weights_imag)
        imag_part = torch.einsum("bntx,tfx->bnfx", input_real, weights_imag) + torch.einsum("bntx,tfx->bnfx", input_imag, weights_real)
        return real_part, imag_part

    def forward(self, x):
        batch_size = x.shape[0]
        # 傅里叶变换
        x_ft = torch.fft.rfft(x.transpose(-1, -2))
        # 获取实部和虚部
        x_real = x_ft.real
        x_imag = x_ft.imag
        # Koopman算子时间推进
        out_real, out_imag = self.time_marching(x_real, x_imag, self.koopman_matrix_real, self.koopman_matrix_imag)
        # 反傅里叶变换
        out_ft = torch.complex(out_real, out_imag)
        x_out = torch.fft.irfft(out_ft, n=x.size(-2))
        return x_out.transpose(-1, -2)


class MultiHeadLocalAttention(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size=3):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        # 局部卷积注意力门控
        self.conv_q = Conv1d(d_model, d_model, kernel_size=kernel_size, padding='same')
        self.conv_k = Conv1d(d_model, d_model, kernel_size=kernel_size, padding='same')
        self.v_linear = nn.Linear(d_model, d_model)
        # 突发事件响应门
        self.event_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x, event_mask=None):
        # x: [B, N, T, C]
        B, N, T, C = x.shape
        x_t = x.transpose(2, 3)  # [B, N, C, T]

        # 局部卷积特征提取
        q = self.conv_q(x_t).transpose(2, 3)  # [B, N, T, C]
        k = self.conv_k(x_t).transpose(2, 3)
        v = self.v_linear(x)

        # 多头划分
        q = q.view(B, N, T, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)  # [B, h, N, T, d_k]
        k = k.view(B, N, T, self.n_heads, self.d_k).permute(0, 3, 1, 4, 2)
        v = v.view(B, N, T, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)

        # 因果注意力计算
        attn = torch.matmul(q, k) / np.sqrt(self.d_k)

        # 创建一个上三角矩阵遮罩
        attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=attn.device), diagonal=1)  # [T, T]
        attn = attn.masked_fill(attn_mask, -float('inf'))
        attn = F.softmax(attn, dim=-1)


        # 突发事件门控增强
        if event_mask is not None:
            event_weights = self.event_gate(x)  # [B, N, T, 1]
            attn = attn * event_mask.unsqueeze(1).unsqueeze(2) + event_weights.transpose(2, 3)

        return torch.matmul(attn, v).transpose(1, 2).reshape(B, N, T, C)

class LongtermForcast(nn.Module):
    def __init__(self,config, seq_len,input_hidden, output_hidden, tcn_layers, revin=False):
        super().__init__()

        self.input_hidden = input_hidden
        self.output_hidden = output_hidden
        self.seq_len = seq_len

        self.kernel_size = config['kernel_size']
        #Dropout类型	建议范围	调整策略
        #head_dropout	0.2-0.5	随模型深度增加而提高
        #config['dropout']	0.1-0.3	输入特征复杂度正相关
        self.dropout = config['dropout']
        self.head_dropout = config['head_dropout']
        self.depth = tcn_layers

        self.DSC_blocks = nn.ModuleList([ImprovedDSCLayer(input_dim=self.input_hidden, kernel_size=self.kernel_size) for _ in range(self.depth)])
        self.W_P = nn.Linear(self.input_hidden, self.output_hidden)
        self.head0 = nn.Sequential(
            nn.Linear(self.input_hidden, self.input_hidden),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.seq_len * self.input_hidden, self.seq_len *self.input_hidden ),
            nn.GELU(),
            #对展平后的高维特征（维度[BN, TC_in]）进行随机屏蔽
            nn.Dropout(self.head_dropout),
            nn.Linear(self.seq_len *self.input_hidden, self.input_hidden * self.seq_len),
            #通过随机丢弃缓解突发流量尖峰导致的梯度爆炸
            nn.Dropout(self.head_dropout)
        )
        #全局Dropou输入噪声注入：在RevIN归一化后增加数据扰动 模拟传感器误差：提升模型对数据采集噪声的鲁棒性
        self.dropout = nn.Dropout(self.dropout)
        self.revin = RevIN(self._num_node) if revin else None

    def forward(self, x_emd):
        bs, num_node, _, _ = x_emd.shape
        if self.revin:
            x_emd = self.revin(x_emd, 'norm')

        #   分支1 (head0)	Linear + GELU	全局特征捕捉
        #   分支2	多层DSC卷积	局部时序模式提取
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

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden



