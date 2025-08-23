class SpectralFilter(nn.Module):
    def __init__(self, F_in, F_out):
        super().__init__()
        self.F_in  = F_in
        self.F_out = F_out
        # 独立线性层，带偏置以增强差异
        self.linear_real = nn.Linear(F_in, F_out, bias=True)
        self.linear_imag = nn.Linear(F_in, F_out, bias=True)
        # 非线性激活
        self.act = nn.GELU()

        # 参数初始化：高斯分布，std 随输出维度缩放
        nn.init.normal_(self.linear_real.weight, std=1.0/np.sqrt(F_out))
        nn.init.normal_(self.linear_imag.weight, std=1.0/np.sqrt(F_out))
        nn.init.zeros_(self.linear_real.bias)
        nn.init.zeros_(self.linear_imag.bias)

    def forward(self, x_spec_segment):
        # x_spec_segment: complex [B, M, T, F_in]
        real = x_spec_segment.real.reshape(-1, self.F_in)
        imag = x_spec_segment.imag.reshape(-1, self.F_in)

        real_out = self.act(self.linear_real(real))
        imag_out = self.act(self.linear_imag(imag))

        B, M, T, _ = x_spec_segment.shape
        real_out = real_out.view(B, M, T, self.F_out)
        imag_out = imag_out.view(B, M, T, self.F_out)
        return torch.complex(real_out, imag_out)


class SpecGraphFreqNet(nn.Module):
    def __init__(self,
                 in_channels, hidden_dim,
                 energy_splits=(0.8,0.95),
                 gate_hidden=16):
        super().__init__()
        self.F_in      = in_channels
        self.hidden_dim= hidden_dim
        self.low_cut, self.mid_cut = energy_splits

        # 四段独立滤波器
        self.filter_high = SpectralFilter(in_channels, hidden_dim)
        self.filter_mid  = SpectralFilter(in_channels, hidden_dim)
        self.filter_low  = SpectralFilter(in_channels, hidden_dim)
        self.filter_neg  = SpectralFilter(in_channels, hidden_dim)

        # 复合门控网络：基于每个 lambda 值生成四段权重
        # 输入维度 1 → 隐藏 → 输出 4 → softmax 得到 [w_high, w_mid, w_low, w_neg]
        self.gate_net = nn.Sequential(
            nn.Linear(1, gate_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(gate_hidden, 4, bias=True),
        )

        # 最终融合投影回时域特征
        self.proj = nn.Linear(hidden_dim, in_channels)

    def forward(self, x, eigvecs, lambdas):
        """
        x: [B, N, T, F_in]
        eigvecs: [N, K] or complex
        lambdas: [K]   频谱索引对应的特征
        """
        B, N, T,_ = x.shape
        device = x.device

        # 1. GFT + FFT → x_spec [B, K, T, F]
        if torch.is_complex(eigvecs):
            x_gft = torch.einsum('bntf, nk -> bktf',
                                 x.to(dtype=torch.complex64),
                                 eigvecs.conj())
        else:
            x_gft = torch.einsum('bntf, nk -> bktf', x, eigvecs)
        x_spec = torch.fft.fft(x_gft, dim=2)

        # 2. 计算能量并拆分索引（保留用于可视化或对比）
        energy = (x_spec.abs()**2).sum(dim=(0,2,3))  # [K]
        neg_mask = lambdas < 0
        pos_idxs = (~neg_mask).nonzero().squeeze()
        pos_sorted = pos_idxs[energy[pos_idxs].argsort(descending=True)]
        k_pos = pos_sorted.numel()
        cut1, cut2 = int(self.low_cut*k_pos), int(self.mid_cut*k_pos)
        high_idx = pos_sorted[:cut1]
        mid_idx  = pos_sorted[cut1:cut2]
        low_idx  = pos_sorted[cut2:]
        neg_idx = (lambdas < 0).nonzero(as_tuple=False).flatten()

        # 3. 软门控权重：每个频谱 lambda 都有一个四段权重
        lam = lambdas.view(-1,1)                            # [K,1]
        gates = self.gate_net(lam)                          # [K,4]
        gates = F.softmax(gates, dim=-1)                    # [K,4]
        w_high, w_mid, w_low, w_neg = gates.unbind(-1)      # 各自 [K]

        # 4. 按段分别应用滤波器，乘以对应 gate 权重后累加
        # 初始化 y_spec
        K = x_spec.shape[1]
        y_spec = torch.zeros([B, K, T, self.hidden_dim], device=x_spec.device, dtype=x_spec.dtype)
        # 对每一段做变换并加权
        if len(high_idx)>0:
            seg = x_spec[:, high_idx]                        # [B, Mh, T, F]
            out = self.filter_high(seg)                      # [B, Mh, T, hidden_dim]
            y_spec[:, high_idx] += out * w_high[high_idx].view(1,-1,1,1)

        if len(mid_idx)>0:
            seg = x_spec[:, mid_idx]
            out = self.filter_mid(seg)
            y_spec[:, mid_idx] += out * w_mid[mid_idx].view(1,-1,1,1)

        if len(low_idx)>0:
            seg = x_spec[:, low_idx]
            out = self.filter_low(seg)
            y_spec[:, low_idx] += out * w_low[low_idx].view(1,-1,1,1)

        if len(neg_idx)>0:
            seg = x_spec[:, neg_idx]
            out = self.filter_neg(seg)
            y_spec[:, neg_idx] += out * w_neg[neg_idx].view(1,-1,1,1)

        # 5. IFFT + IGFT → 回到时域
        y_ifft = torch.fft.ifft(y_spec, dim=2)
        if not torch.is_complex(eigvecs):
            y_ifft = y_ifft.real

        y_rec = torch.einsum('bktf, nk -> bntf', y_ifft, eigvecs)
        # 6. 最后投影到输入维度并残差连接
        y_out = self.proj(y_rec.real)

        return y_out
