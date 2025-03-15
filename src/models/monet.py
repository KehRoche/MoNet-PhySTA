import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from src.base.model import BaseModel
from .TempEvo import TempEvo
from .SpitalDif import SpitalDif
from .UniMoudle import *



class MoNet(BaseModel):
    def __init__(self, input_dim,output_dim,model_config):
        super(MoNet, self).__init__(input_dim,output_dim)
        self.input_dim = output_dim

        self.A = model_config['adj']
        self.L = compute_laplacian(self.A)
        self.num_nodes = model_config['num_nodes']

        batch_size = model_config['bs']
        seq_len = model_config['seq_len']
        emd_dim = model_config['emd_dim']

        hidden_dim = model_config['hidden_dim']
        tcn_layers = model_config['tcn_layers']
        kno_layers = model_config['kno_layers']

        #model_config['covariate_dim']

        self.location = model_config['location']
        self.location = self.location.unsqueeze(0).unsqueeze(2)
        self.location = self.location.repeat(batch_size,1,seq_len,1).float()

        # 计算正弦值
        sin_wave = torch.sin(torch.linspace(0, 2 * np.pi, seq_len))
        global_feature_init = sin_wave.unsqueeze(-1)
        # 生成与global_feature_init相同形状的随机噪声
        noise = torch.randn_like(global_feature_init) * 0.1  # 0.1是噪声的标准差
        global_feature = global_feature_init + noise
        
        #tod,dow,node,location,common_feat
        num_conditon = 3
        condition_emb = model_config['condition_emb']
        self.hidden_dim = condition_emb*num_conditon+emd_dim

        #新的embeding方法
        # spatial embeddings
        self.node_emb = nn.Parameter(
            torch.empty(self.num_nodes, condition_emb))
        nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        #两种不同的时间信息编码方式，将每个时刻的特性以emd_dim向量描述，并在编码时将最近时刻的emd向量取出并替换
        self.time_in_day_emb = nn.Parameter(
            torch.empty(288, condition_emb))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
            torch.empty(7, condition_emb))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        #self.com_fea = nn.Parameter(seq_len,self.num_nodes,self.input_dim)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=3 * seq_len, out_channels=emd_dim*seq_len, kernel_size=(1, 1), bias=True)


        self.global_feature = nn.Parameter(global_feature)
        self.dyn_gate = DynmiacGate(self.input_dim,emd_dim)
        self.period_fun = nn.Linear(emd_dim*2, emd_dim)
        #embedding param
        self.emb_way = model_config['emb_way']
        self.emd_dim = emd_dim
        self.data_embedding = nn.Linear(self.input_dim, emd_dim)
        self.tod_embedding = nn.Linear(self.input_dim, emd_dim//2)
        self.dow_embedding = nn.Linear(self.input_dim, emd_dim//2)
        self.loaction_embedding = nn.Linear(self.input_dim*3, condition_emb)
        self.fusion_embedding = nn.Linear(self.emd_dim*3, emd_dim)

        #path
        from .PhyField import SpectralFusionLayer

        self.Field = SpectralFusionLayer(self.hidden_dim,self.hidden_dim,12,6,self.A)
        #self.TempModule = TempEvo(model_config,self.input_dim,seq_len,self.hidden_dim, tcn_layers,kno_layers)
        #self.SptialModule= SpitalDif(model_config,self.input_dim,seq_len,self.hidden_dim)

        self.activation = nn.ReLU()

        #output_fusion
        self.output_fusion = nn.Sequential(
            nn.Linear(self.input_dim*3, self.input_dim))
            # ,
            # self.activation,
            # nn.Linear(emd_dim//2, 1))
        # self.router_weight = nn.Parameter(torch.zeros(1, 1,seq_len,emd_dim), requires_grad=True)
        # self.out_fc_1   = nn.Linear(emd_dim, emd_dim//2)
        # self.out_fc_2   = nn.Linear(emd_dim//2, self.input_dim)

    def STIDemd(self,input):

        time_in_day_emb = self.time_in_day_emb[(input[:, :, :,self.input_dim] * 288).type(torch.LongTensor)]
        day_in_week_emb = self.day_in_week_emb[(input[:, :, :,self.input_dim+1] * 7).type(torch.LongTensor)]

        #xyz = self.loaction_embedding(location)
        # time series embedding
        batch_size, seq_len, num_nodes, _ = input.shape
        input = input.transpose(1, 2).contiguous()
        input = input.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input).squeeze(-1).view(batch_size, self.emd_dim,num_nodes,-1)

        #common_feat
        # com_fea = self.com_fea.unsqueeze(0)
        # com_fea = com_fea.repeat(batch_size,1,1,1).float()
        # irm_fea = self.time_series_emb_layer(torch.cat([com_fea,tem_emb],dim=1))


        node_emb = []
        # expand node embeddings
        node_emb.append(self.node_emb.unsqueeze(0).expand(
            batch_size, -1, -1).transpose(1, 2).unsqueeze(-1).repeat(1, 1, 1,seq_len))

        # temporal embeddings
        tem_emb = []
        tem_emb.append(time_in_day_emb.transpose(1, 3))
        tem_emb.append(day_in_week_emb.transpose(1, 3))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        #b,emd*4,nodes,feat,
        #b,feat,nodes,seq_len
        return hidden.view(batch_size, seq_len*self.hidden_dim,num_nodes,-1),hidden
    def embedding(self,input,time,location,laplacian,embway="SOP"):
        tod = self.tod_embedding(time[:,:,:,:1])
        dow = self.dow_embedding(time[:, :, :, 1:])
        xyz = self.loaction_embedding(location)
        condition_info = self.activation(torch.concat((tod, dow, xyz), dim=-1))
        input = self.data_embedding(input)
        input = self.fusion_embedding(torch.cat((input,condition_info), dim=-1))
        #emb_fea = self.dyn_gate(input,condition_info)


        # if('P' in embway):
        #     global_fea = self.global_feature.unsqueeze(0).unsqueeze(0)
        #     global_fea = global_fea.repeat(input.shape[0], input.shape[1], 1,1)
        #     global_fea = self.data_embedding(global_fea)
        #     #依赖共性变量学习周期性特征
        #     common_fea = self.period_fun(torch.concat([global_fea, tod,dow], dim=-1))
        #     emb_fea = emb_fea + common_fea
        # if('S' in embway):
        #     # 计算拉普拉斯矩阵的特征值和特征向量
        #     eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        #     # 选择前 k 个最小的非平凡特征值对应的特征向量
        #     # 排除第一个特征值（通常为零），选择其余的 k 个最小特征值对应的特征向量
        #     x_spe = eigenvectors[:, 1:self.emd_dim + 1].unsqueeze(1)
        #     x_spe = x_spe.unsqueeze(0).repeat(input.shape[0],1,input.shape[2],1)
        #     emb_fea = emb_fea + x_spe
        return input
    def forward(self, input,  label=None):
        #batch,len,nodes,feat
        X = input[:,:,:,:self.input_dim]
        time = input[:,:,:,self.input_dim:]
        #X = self.embedding(X,time,self.location,self.L,self.emb_way)
        #b,feat,nodes,len
        mix_X,X = self.STIDemd(input)
        X_phy = self.Field(X)
        #todo : 编码后x的输入为b,64,325,1有问题
        #X_inevo = self.TempModule(mix_X,time).transpose(1, 2)
        #_exdif = self.SptialModule(X,self.A,time)
        #,X_exdif，X_inevo,X_phy
        # y_hat = self.output_fusion(torch.cat((X_inevo,X_phy,X_exdif),dim=-1))
        #+ X_inevo + X_exdif
        y_hat = X_phy
        #y_hat = self.output_fusion(X_inevo)


        # forecast    = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        # forecast    = forecast.transpose(1,2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        return  y_hat.transpose_(1, 2)
        #return X_inevo








