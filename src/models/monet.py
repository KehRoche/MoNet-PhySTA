import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from src.base.model import BaseModel
from .PhyField import GTFNO2d
from .SpitalDif import MSKGN
from .UniMoudle import *



class MoNet(BaseModel):
    def __init__(self, input_dim,output_dim,model_config):
        super(MoNet, self).__init__(input_dim,output_dim)
        self.corvar_dim = self.input_dim-4
        self.output_dim = output_dim

        self.A = model_config['adj']
        self.eigvecs,self.eigval = compute_laplacian(self.A)
        self.num_nodes = self.A.shape[0]


        batch_size = model_config['bs']
        seq_len = model_config['seq_len']
        emd_dim = model_config['emd_dim']
        #cosl
        self.gfno_hidden = model_config['gfno_hidden']
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

        if self.corvar_dim >1:
            self.time_interval = 8
            num_conditon = 5
            self.fea_dim = input_dim - 4
            self.side_encoding = nn.Sequential(
                nn.Conv2d(in_channels=self.corvar_dim * seq_len, out_channels=emd_dim*seq_len, kernel_size=(1, 1), bias=True),
                self.activation,
                nn.Dropout(p=0.15),
                MultiLayerPerceptron(emd_dim*seq_len, emd_dim*seq_len)
            )

        #tod,dow,node,location,common_feat
        #embedding param
        self.data_embedding = nn.Linear(1, emd_dim)
        self.tod_embedding = nn.Linear(1, emd_dim//2)
        self.dow_embedding = nn.Linear(1, emd_dim//2)
        self.fusion_embedding = nn.Linear(self.emd_dim*3, emd_dim)

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


        self.fusion_embedding = nn.Linear(self.emd_dim*2, emd_dim)

        #path
        self.Field = GTFNO2d(x=self.num_nodes,t = seq_len,width = self.emd_dim)
        self.SptialModule = MSKGN(hidden_channels=self.hidden_dim, levels=self.graph_layers,adj_matrix=self.A,topk=self.topk_edges)

        #output_fusion
        self.output_fusion = nn.Sequential(
            #phy,sptial,side_embeding
            nn.Linear(self.hidden_dim//2+(self.emd_dim if self.corvar_dim > 1 else 0)+self.emd_dim, self.emd_dim),
            self.activation,
            nn.Dropout(p=0.15),
            nn.Linear(self.emd_dim, self.output_dim))
        self.res_layer = MultiLayerPerceptron(self.hidden_dim, seq_len)



    def STIDemd(self,input):
        assert input[:, :, :, 2].max() < 1,input[:, :, :, 2].max().item() # 检查是否略大于 1.0
        assert input[:, :, :, 1].max() < 1,input[:, :, :, 1].max().item()

        time_in_day_emb = self.time_in_day_emb[(input[:, -1:, :,1] * self.time_interval).type(torch.LongTensor)]
        day_in_week_emb = self.day_in_week_emb[(input[:, -1:, :,2] * 7).type(torch.LongTensor)]

        tod_full_emb = self.time_in_day_emb[(input[:, :, :,1] * self.time_interval).type(torch.LongTensor)]
        dow_full_emb = self.day_in_week_emb[(input[:,:, :,2] * 7).type(torch.LongTensor)]
        if self.corvar_dim >1:
            day_in_month_emb = self.day_in_month_emb[(input[:, -1:, :,3] * 31).type(torch.LongTensor)]
            month_in_year_emb = self.month_in_year_emb[(input[:, -1:, :,4] * 12).type(torch.LongTensor)]

        # time series embedding
        batch_size, seq_len, num_nodes, _ = input.shape
        input = input.transpose(1, 2).contiguous()
        input = input.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input).squeeze(-1).view(batch_size, self.emd_dim,num_nodes,-1)

        node_emb = []
        # expand node embeddings
        node_emb.append(self.node_emb.unsqueeze(0).expand(
            batch_size, -1, -1).transpose(1, 2).unsqueeze(-1).repeat(1, 1, 1,1))

        # temporal embeddings
        tem_emb = []
        tem_emb.append(time_in_day_emb.transpose(1, 3))
        tem_emb.append(day_in_week_emb.transpose(1, 3))


        tem_full_emb = []
        tem_full_emb.append(tod_full_emb.transpose(1, 3))
        tem_full_emb.append(dow_full_emb.transpose(1, 3))

        if self.corvar_dim >1:
            tem_emb.append(day_in_month_emb.transpose(1, 3))
            tem_emb.append(month_in_year_emb.transpose(1, 3))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # list -> tensor（在通道维拼接）; 保持 dtype/device 一致
        tem_full_emb = torch.cat(
            [x.to(device=time_series_emb.device, dtype=time_series_emb.dtype)
             for x in tem_full_emb],
            dim=1,  # [B, C_tem, N, T]
        ).mean(dim=2)



        #b,emd*4,nodes,feat,
        #b,feat,nodes,seq_len
        return hidden.view(batch_size, self.hidden_dim,num_nodes,-1),tem_full_emb
    def embedding(self,input,time):
        tod = self.tod_embedding(time[:,:,:,0:1])
        dow = self.dow_embedding(time[:, :, :, 1:2])


        #xyz = self.loaction_embedding(location)
        condition_info = self.activation(torch.concat((tod, dow), dim=-1))
        input = self.data_embedding(input)
        #input = self.fusion_embedding(torch.cat((input,condition_info), dim=-1))
        #emb_fea = self.dyn_gate(input,condition_info)
        return input
    def forward(self, input,  label=None):
        #batch,len,nodes,feat
        batch,len,nodes,feat = input.shape
        fea = input[:,:,:,:1]
        time = input[:,:,:,self.fea_dim:]

        #X = self.embedding(fea,time)
        mix_X,time_feats = self.STIDemd(torch.concat([fea,time],dim=-1))
        #b,feat,nodes,len x_phy:b,n,l,f
        X_phy = self.Field(fea.transpose(1,2),self.eigvecs,self.eigval,time_feats).transpose(1,2)
        X_exdif = self.SptialModule(mix_X,self.A).repeat(1,1,1,len).transpose(1,3)
        x_res = self.res_layer(mix_X)
        if self.corvar_dim > 0:
            convar = input[:,:,:,1:self.corvar_dim+1]
            x_side = self.side_encoding(convar.reshape(batch,-1,nodes,1))
            x_side = x_side.reshape(batch,len,nodes,-1)
            #x_res = x_res + x_side
            y_hat = self.output_fusion(torch.cat((X_phy,X_exdif,x_side),dim=-1))+self.activation(x_res)
        else:
            y_hat = self.output_fusion(torch.cat((X_phy,X_exdif),dim=-1))+self.activation(x_res)
        return y_hat

