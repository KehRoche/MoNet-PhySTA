import torch
import torch.nn as nn
from .UniMoudle import *



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

