import numpy as np
import torch
import torch.nn as nn
from sympy.abc import alpha


class SpectralFilter(nn.Module):
    def __init__(self, F_in, F_out):
        super().__init__()
        self.F_in  = F_in
        self.F_out = F_out
        # 分别为实部和虚部设计独立的线性层
        self.linear_real = nn.Linear(F_in, F_out, bias=False)
        self.linear_imag = nn.Linear(F_in, F_out, bias=False)

    def forward(self, x_spec_segment):
        """
        Args:
          x_spec_segment: complex tensor of shape [B, M, T, F_in]
        Returns:
          complex tensor of shape [B, M, T, F_out]
        """
        # 拆分实部和虚部
        real = x_spec_segment.real      # [B, M, T, F_in]
        imag = x_spec_segment.imag      # [B, M, T, F_in]

        B, M, T, _ = real.shape

        # 合并前 3 维以便用线性层批量映射
        real_flat = real.reshape(-1, self.F_in)   # [(B*M*T), F_in]
        imag_flat = imag.reshape(-1, self.F_in)   # [(B*M*T), F_in]

        # 分别映射
        real_out = self.linear_real(real_flat)    # [(B*M*T), F_out]
        imag_out = self.linear_imag(imag_flat)    # [(B*M*T), F_out]

        # 恢复形状
        real_out = real_out.view(B, M, T, self.F_out)
        imag_out = imag_out.view(B, M, T, self.F_out)

        # 重新组装成复数张量并返回
        return torch.complex(real_out, imag_out)


class SpecGraphFreqNet(nn.Module):
    def __init__(self,
                 in_channels, hidden_dim, energy_splits=(0.8,0.95)):
        super().__init__()
        self.F_in = in_channels
        self.hidden_dim = hidden_dim
        self.low_cut, self.mid_cut = energy_splits

        # 四段 SpectralFilter，但内部全共享 FNO 风格线性层
        self.filter_high = SpectralFilter(in_channels, hidden_dim)
        self.filter_mid  = SpectralFilter(in_channels, hidden_dim)
        self.filter_low  = SpectralFilter(in_channels, hidden_dim)
        self.filter_neg  = SpectralFilter(in_channels, hidden_dim)

        self.proj   = nn.Linear(hidden_dim, in_channels)

    def forward(self, x, eigvecs, lambdas):
        B, N, T, F = x.shape
        device = x.device
        K = eigvecs.size(1)
        # 执行 GFT
        # 1. GFT & FFT 如前...
        if torch.is_complex(eigvecs):
            x_gft = torch.einsum('bntf, nk -> bktf', x.to(dtype=torch.complex64), eigvecs.conj())  # [B, K, T, F]
        else:
            x_gft = torch.einsum('bntf, nk -> bktf', x, eigvecs)  # [B, K, T, F]

        x_spec = torch.fft.fft(x_gft, dim=2)                       # [B, K, T, F]

        # 2. 能量分段索引
        energy = (x_spec.abs()**2).sum(dim=(0,2,3))       # [K]
        neg_mask = lambdas < 0
        pos_idxs = (~neg_mask).nonzero().squeeze()
        pos_sorted = pos_idxs[energy[pos_idxs].argsort(descending=True)]
        k_pos = pos_sorted.numel()
        cut1, cut2 = int(self.low_cut*k_pos), int(self.mid_cut*k_pos)
        high_idx = pos_sorted[:cut1]
        mid_idx  = pos_sorted[cut1:cut2]
        low_idx  = pos_sorted[cut2:]
        neg_idx  = neg_mask.nonzero().squeeze()

        # 3. 分段提取子张量并应用对应滤波器
        y_spec = torch.zeros([B,N,x_spec.shape[2],self.hidden_dim], device=device).to(dtype=x_spec.dtype)
        if len(high_idx)>0:
            seg = x_spec[:, high_idx]            # [B, M_high, T, F]
            y_spec[:, high_idx] = self.filter_high(seg)
        if len(mid_idx)>0:
            seg = x_spec[:, mid_idx]
            y_spec[:, mid_idx]  = self.filter_mid(seg)
        if len(low_idx)>0:
            seg = x_spec[:, low_idx]
            y_spec[:, low_idx]  = self.filter_low(seg)
        if torch.is_complex(eigvecs) and len(neg_idx)>0:
            seg = x_spec[:, neg_idx]
            y_spec[:, neg_idx]  = self.filter_neg(seg)

        # 4. 逆 FFT & 逆 GFT 如前...
        y_ifft = torch.fft.ifft(y_spec, dim=2)
        if not torch.is_complex(eigvecs):
            y_ifft = y_ifft.real
        y_rec  = torch.einsum('bktf, nk -> bntf', y_ifft, eigvecs)

        # 5. 输出投影
        return self.proj(y_rec.real)

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

    def forward(self, x,eigvecs):
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