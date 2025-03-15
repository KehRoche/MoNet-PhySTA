import numpy as np
import torch
import torch.nn as nn
from sympy.abc import alpha


def build_directed_laplacian(A):
    out_degree = torch.sum(A, dim=1)          # 出度矩阵对角线元素
    D_out = torch.diag(out_degree)            # 出度对角矩阵
    L = D_out - A                             # 出度拉普拉斯矩阵
    return L


def directed_gft(L, x):
    U, S, V = torch.svd(L)
    x_hat_fwd = torch.matmul(U.T, x)        # 正向频域投影
    x_hat_bwd = torch.matmul(V.T, x)       # 逆向频域投影
    return x_hat_fwd, x_hat_bwd

def inverse_directed_gft(U, V, x_hat_fwd, x_hat_bwd):
    x_recon_fwd = torch.matmul(U, x_hat_fwd)
    x_recon_bwd = torch.matmul(V, x_hat_bwd)
    return x_recon_fwd, x_recon_bwd

def calu_A_dir(adj_matrix):
    # 计算方向差异矩阵
    diff_matrix = adj_matrix - adj_matrix.T

    # 定义显著性阈值（基于差异的绝对值）
    threshold = torch.quantile(diff_matrix[diff_matrix != 0].abs(), 0.25)  # 取差异绝对值的第75百分位数

    # 生成二进制方向矩阵（1表示i→j方向显著）
    dir_mask = (diff_matrix > threshold).float()

    # 根据原生权重增强方向矩阵（例如，仅保留高权重的方向）
    weight_threshold = torch.median(adj_matrix[adj_matrix != 0]) / 4
    enhanced_dir_matrix = dir_mask * (adj_matrix > weight_threshold).float()

    return enhanced_dir_matrix


class SpectralFusionLayer(nn.Module):
    def __init__(self, in_feat, out_feat, graph_modes, time_modes,A):
        super().__init__()
        self.graph_modes = graph_modes  # 保留的图频域模式数
        self.time_modes = time_modes  # 保留的时频域模式数
        alpha = 0.2
        A_dir = calu_A_dir(A)
        A_complex = torch.complex(
            A * alpha,
            A_dir * (1 - alpha)
        )
        D_out = torch.diag_embed(torch.sum(A_complex, dim=1))
        # 复数拉普拉斯矩阵
        L_complex = D_out - A_complex
        # 强制埃尔米特化以稳定分解
        L_herm = (L_complex + L_complex.conj().T) / 2
        eigenvalues, eigenvectors = torch.linalg.eigh(L_herm.real)
        # 按特征值实部排序（能量降序）
        _, indices = torch.sort(torch.real(eigenvalues), descending=False)
        self.lap_evecs = eigenvectors[:, indices[:self.graph_modes]]

        # 初始化象限权重参数（实部+虚部分开初始化）
        self.weights_quad1 = nn.Parameter(torch.randn(
            in_feat, out_feat, graph_modes, time_modes, 2))
        self.weights_quad2 = nn.Parameter(torch.randn(
            in_feat, out_feat, graph_modes, time_modes, 2))
        self.weights_quad3 = nn.Parameter(torch.randn(
            in_feat, out_feat, graph_modes, time_modes, 2))
        self.weights_quad4 = nn.Parameter(torch.randn(
            in_feat, out_feat, graph_modes, time_modes, 2))
        
        self.output = nn.Linear(out_feat, 1)

    def compl_mul(self, x_ft, weights):
        # 复数乘法分解为实部虚部分别计算 (batch, in, modes, modes) * (in, out, modes, modes)
        real = torch.einsum('bintf, ionth->bontf', x_ft.real, weights[..., :1]) \
               - torch.einsum('bintf, ionth->bontf', x_ft.imag, weights[..., 1:])
        imag = torch.einsum('bintf, ionth->bontf', x_ft.real, weights[..., :1]) \
               + torch.einsum('bintf, ionth->bontf', x_ft.imag, weights[..., 1:])
        return torch.view_as_complex(torch.stack([real, imag], dim=-1))

    def forward(self, x):
        # x shape: (batch, nodes, time, feat)
        x = x.permute(0, 2, 3,1).contiguous()
        batch_size = x.shape[0]

        # -- Step 1: 图傅里叶变换(GFT) --#
        # 沿nodes维度投影到图频域 (batch, nodes, time, feat) -> (batch, modes, time, feat)
        #torch.einsum('nq, bntf -> bqtf', self.lap_evecs[:, :self.graph_modes], x)
        x_gft = torch.einsum('nq,bntf->bqtf', self.lap_evecs.conj(), x)

        # -- Step 2: 时间傅里叶变换(FFT) --#
        # 沿时间维度进行FFT (batch, modes, time, feat) -> (batch, modes, freq, feat, complex)
        x_ft = torch.fft.rfft(x_gft.unsqueeze(-1), dim=2)
        x_ft= x_ft.permute(0,3,1,2,4)

        freq_len = x_ft.shape[2]

        # -- Step 3: 频域象限划分与参数化滤波 --#
        # 划分四个象限（示例：前1/2图低频+前1/2时低频为Quad1，其他类推）
        quad1 = x_ft[:, :, :self.graph_modes, :self.time_modes, :]
        quad2 = x_ft[:, :, :self.graph_modes, -self.time_modes:, :]
        quad3 = x_ft[:, :, -self.graph_modes:, :self.time_modes, :]
        quad4 = x_ft[:, :, -self.graph_modes:, -self.time_modes:, :]

        # 对每个象限应用不同权重
        out_ft = torch.zeros(batch_size, self.weights_quad1.shape[1],
                             self.graph_modes, freq_len//2+1, 1,
                             dtype=torch.complex64, device=x.device)

        out_ft[:, :, :self.graph_modes, :self.time_modes] = self.compl_mul(
            quad1, self.weights_quad1)
        out_ft[:, :, :self.graph_modes, -self.time_modes:] = self.compl_mul(
            quad2, self.weights_quad2)
        out_ft[:, :, -self.graph_modes:, :self.time_modes] = self.compl_mul(
            quad3, self.weights_quad3)
        out_ft[:, :, -self.graph_modes:, -self.time_modes:] = self.compl_mul(
            quad4, self.weights_quad4)

        out_ft = out_ft.permute(0,2,3,1,4)
        # -- Step 4: 逆变换恢复时空域 --#
        # 逆FFT (batch, modes, freq, feat) -> (batch, modes, time, feat)
        x_ifft = torch.fft.irfft(out_ft.squeeze(-1), n=x_gft.shape[2], dim=2)

        # 逆GFT (batch, modes, time, feat) -> (batch, nodes, time, feat)
        x_igft = torch.einsum('nq, bqtf -> bntf',
                              self.lap_evecs, x_ifft)
        
        y_phy = self.output(x_igft)
        return y_phy