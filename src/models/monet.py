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
        seq_len = model_config['seq_len']
        emd_dim = model_config['emd_dim']
        hidden_dim = model_config['hidden_dim']
        tcn_layers = model_config['tcn_layers']
        kno_layers = model_config['kno_layers']
        #model_config['covariate_dim']
        self.emb_dim = emd_dim
        self.data_embedding = nn.Linear(self.input_dim, emd_dim)
        self.tod_embedding = nn.Linear(self.input_dim, emd_dim)
        self.dow_embedding = nn.Linear(self.input_dim, emd_dim)

        # deterministic path
        self.TempModule = TempEvo(model_config,self.input_dim,seq_len,hidden_dim, tcn_layers,kno_layers)

        self.SptialModule= SpitalDif(model_config,self.input_dim,seq_len,hidden_dim)

        self.router_weight = nn.Parameter(torch.zeros(1, 1,seq_len,emd_dim), requires_grad=True)
        self.out_fc_1   = nn.Linear(emd_dim, emd_dim//2)
        self.out_fc_2   = nn.Linear(emd_dim//2, self.input_dim)

    def embedding(self,input,time,laplacian):
        # 计算拉普拉斯矩阵的特征值和特征向量
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        # 选择前 k 个最小的非平凡特征值对应的特征向量
        # 排除第一个特征值（通常为零），选择其余的 k 个最小特征值对应的特征向量
        x_spe = eigenvectors[:, 1:self.emb_dim + 1].unsqueeze(1)
        x_spe = x_spe.unsqueeze(0).repeat(input.shape[0],1,input.shape[2],1)
        x_data = self.data_embedding(input)
        x_time = self.tod_embedding(time[:,:,:,:1])+self.dow_embedding(time[:,:,:,1:])
        return x_spe+x_data+x_time
    def forward(self, input,  label=None):
        #batch,len,nodes,feat
        input.transpose_(1, 2)
        X = input[:,:,:,:self.input_dim]
        time = input[:,:,:,self.input_dim:]
        X = self.embedding(X,time,self.L)
        X_inevo = self.TempModule(X,time)
        X_exdif = self.SptialModule(X,self.A,time)

        # weight_inevo = 0.5*torch.ones_like(X_inevo)+self.router_weight
        # weight_exdif = 0.5*torch.ones_like(X_exdif)-self.router_weight
        # y_hat = weight_inevo*X_inevo + weight_exdif*X_exdif
        y_hat = X_exdif
        y_hat =  F.gelu(self.out_fc_1(y_hat))
        y_hat = self.out_fc_2(y_hat)

        # forecast    = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        # forecast    = forecast.transpose(1,2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        return  y_hat.transpose_(1, 2)








