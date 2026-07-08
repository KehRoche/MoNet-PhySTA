import torch
import torch.nn as nn


class GraphSpectralConvS_Nystrom(nn.Module):
    def __init__(self, in_channels, out_channels, N, modes_t,
                 approximation_rank: int = 256,
                 bias=False,
                 device=None,
                 dtype=torch.float32,
                 debug=False):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.N = int(N)
        self.modes_t = int(modes_t)
        self.device = device
        self.dtype = dtype
        self.debug = debug

        self.m = min(approximation_rank, N)

        # === 修改：使用实数权重，分别处理实部和虚部 ===
        # 为实部和虚部各定义一组权重
        self.spectral_weights_real = nn.Parameter(
            torch.randn(self.m, self.modes_t, in_channels, out_channels, dtype=torch.float32)
            * (1 / (in_channels * out_channels))
        )
        self.spectral_weights_imag = nn.Parameter(
            torch.randn(self.m, self.modes_t, in_channels, out_channels, dtype=torch.float32)
            * (1 / (in_channels * out_channels))
        )

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('U_approx', None)
        self.valid_m = self.m

    def set_basis(self, U_approx):
        if U_approx.shape[0] != self.N:
            raise ValueError(
                f"U_approx node count mismatch: got {U_approx.shape[0]}, expected {self.N}"
            )

        actual_m = U_approx.shape[1]
        self.valid_m = min(self.m, actual_m)

        self.U_approx = U_approx[:, :self.valid_m].to(self.device)
        if self.debug:
            print(f"[GraphSpectralConvS] Basis set: N={self.N}, m={self.valid_m}")

    def forward(self, x):
        if self.U_approx is None:
            raise RuntimeError("Call set_basis(U_approx) before forward pass.")

        if self.debug:
            print(f"\n[GraphSpectralConvS Forward] Input shape: {x.shape}")

        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got shape {x.shape}")

        # 格式识别
        B = x.shape[0]
        if x.shape[1] == self.N and x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2)  # [B, N, T, C] -> [B, C, N, T]
        elif x.shape[1] == self.in_channels and x.shape[2] == self.N:
            pass
        else:
            raise ValueError(f"Input shape {x.shape} doesn't match expected formats")

        B, Cin, N, T = x.shape

        if Cin != self.in_channels or N != self.N:
            raise ValueError(
                f"Dimension mismatch: got C={Cin}, N={N}, expected C={self.in_channels}, N={self.N}"
            )

        # === 1. 时间维度 FFT ===
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')  # [B, Cin, N, T_freq] 复数
        T_freq = x_fft.shape[-1]

        # 截断时间频率
        eff_modes = min(self.modes_t, T_freq)
        x_fft = x_fft[..., :eff_modes]  # [B, Cin, N, eff_modes]

        # === 2. 分离实部和虚部 ===
        x_fft_real = x_fft.real  # [B, Cin, N, eff_modes]
        x_fft_imag = x_fft.imag  # [B, Cin, N, eff_modes]

        # === 3. 空间投影 (GFT) - 实数运算 ===
        U = self.U_approx  # [N, valid_m] 实数
        weights_real = self.spectral_weights_real[:self.valid_m, :eff_modes, :, :]
        weights_imag = self.spectral_weights_imag[:self.valid_m, :eff_modes, :, :]

        # 投影到谱域: U^T @ x
        # 实部投影
        x_spec_real = torch.einsum('nm, bcnt -> bcmt', U, x_fft_real)  # [B, Cin, valid_m, eff_modes]
        # 虚部投影
        x_spec_imag = torch.einsum('nm, bcnt -> bcmt', U, x_fft_imag)  # [B, Cin, valid_m, eff_modes]

        # === 4. 谱域卷积 - 复数乘法规则 ===
        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        # x_spec = x_spec_real + i * x_spec_imag
        # weights = weights_real + i * weights_imag

        # 实部: ac - bd
        out_real = (
                torch.einsum('bcmt, mtio -> bomt', x_spec_real, weights_real) -
                torch.einsum('bcmt, mtio -> bomt', x_spec_imag, weights_imag)
        )

        # 虚部: ad + bc
        out_imag = (
                torch.einsum('bcmt, mtio -> bomt', x_spec_real, weights_imag) +
                torch.einsum('bcmt, mtio -> bomt', x_spec_imag, weights_real)
        )

        # === 5. 空间逆投影 (IGFT) ===
        # U @ x_spec
        x_spatial_real = torch.einsum('nm, bomt -> bont', U, out_real)  # [B, Cout, N, eff_modes]
        x_spatial_imag = torch.einsum('nm, bomt -> bont', U, out_imag)  # [B, Cout, N, eff_modes]

        # === 6. 合并为复数张量 ===
        x_spatial_complex = torch.complex(x_spatial_real, x_spatial_imag)

        # === 7. 时间逆 FFT ===
        # 补零恢复原始时间长度
        if eff_modes < T_freq:
            x_spatial_complex = torch.nn.functional.pad(
                x_spatial_complex, (0, T_freq - eff_modes)
            )

        x_out = torch.fft.irfft(x_spatial_complex, n=T, dim=-1, norm='ortho')  # [B, Cout, N, T]

        if self.bias is not None:
            x_out = x_out + self.bias.view(1, -1, 1, 1)

        if self.debug:
            print(f"  Output shape: {x_out.shape}")

        return x_out


class GraphSpectralConvT(GraphSpectralConvS_Nystrom):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            N: int,
            modes_t: int,
            approximation_rank: int = 256,
            delta: float = 1e-1,
            out_steps: int = None,
            bias: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            N=N,
            modes_t=modes_t,
            approximation_rank=approximation_rank,
            bias=bias,
            **kwargs
        )

        self.out_steps = out_steps
        self.delta = delta


class NyFNO2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            width: int,
            N: int,
            T: int,
            modes_t: int,
            nystrom_rank: int = 64,
            num_spectral_layers: int = 4,
            latent_steps: int = None,
            out_steps: int = None,
            debug: bool = False,
            **kwargs
    ):
        super().__init__()

        self.N = N
        self.width = width
        self.nystrom_rank = nystrom_rank
        self.debug = debug

        # --- Lifting Operator (修复：不再降采样) ---
        self.lifting_operator = LiftingOperator(
            input_channels=in_channels,
            width=width,
            N=N,  # 使用完整的 N，而不是 N//4
            modes_t=modes_t,
            input_shape=(N, T),
            latent_steps=latent_steps,
            nystrom_rank=nystrom_rank,
            debug=debug,
            **kwargs
        )

        # --- 主干谱卷积层 ---
        assert num_spectral_layers > 1
        processing_layers = num_spectral_layers - 1

        self.spectral_conv = nn.ModuleList(
            [
                GraphSpectralConvT(
                    in_channels=width,
                    out_channels=width,
                    N=N,
                    modes_t=modes_t,
                    approximation_rank=nystrom_rank,
                    out_steps=T if out_steps is None else out_steps,
                    bias=True,
                    debug=debug,
                )
                for _ in range(processing_layers)
            ]
        )

        # --- MLPs ---
        self.mlp = nn.ModuleList(
            [MLP(width, width, width) for _ in range(processing_layers)]
        )

        # --- 输出层 ---
        self.final_proj = nn.Conv2d(width, 1, kernel_size=1)

    def inject_basis(self, U_approx):
        """将 Nyström 基注入所有子模块"""
        # 检查维度匹配
        if U_approx.shape[0] != self.N:
            raise ValueError(
                f"U_approx dimension mismatch: got {U_approx.shape[0]} nodes, "
                f"model expects {self.N} nodes"
            )

        # 1. 注入 LiftingOperator
        self.lifting_operator.set_basis(U_approx)

        # 2. 注入主干层
        for idx, layer in enumerate(self.spectral_conv):
            layer.set_basis(U_approx)
            if self.debug:
                print(f"[GTFNO2d] Basis injected into spectral_conv[{idx}]")

        if self.debug:
            print(f"[GTFNO2d] Total: Basis injected into {1 + len(self.spectral_conv)} layers")

    def forward(self, x):
        """
        x: [Batch, N, T, C_in]
        输出: [Batch, 1, N, T]
        """
        # 1. Lifting
        x = self.lifting_operator(x)  # [B, width, N, T]

        # 2. Spectral Layers + Residual
        for sconv, mlp in zip(self.spectral_conv, self.mlp):
            x_in = x

            # 谱卷积分支 (输入输出都是 [B, width, N, T])
            x_spec = sconv(x)

            # MLP 分支
            x_mlp = x.permute(0, 2, 3, 1)  # [B, N, T, width]
            x_mlp = mlp(x_mlp)
            x_mlp = x_mlp.permute(0, 3, 1, 2)  # [B, width, N, T]

            # 融合
            x = x_spec + x_mlp + x_in
            x = torch.nn.functional.gelu(x)

        # 3. 输出
        x = self.final_proj(x)  # [B, 1, N, T]

        return x.permute(0, 2, 3, 1)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [B, N, T, C] or [B, C, N, T] depending on usage
        # 需要确保输入维度在最后一维
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LiftingOperator(nn.Module):
    def __init__(
            self,
            input_channels,
            width,
            N,  # 使用完整的节点数
            modes_t,
            input_shape,
            latent_steps=None,
            norm="ortho",
            beta=0.1,
            activation=nn.GELU(),
            spatial_random_feats=False,
            channel_expansion=1,
            nonlinear=True,
            nystrom_rank=64,
            debug=False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.width = width
        self.N = N
        self.debug = debug

        # 关键修复：sconv 使用完整的 N
        self.sconv = GraphSpectralConvS_Nystrom(
            in_channels=input_channels,
            out_channels=width,
            N=N,  # 不再使用 N//4
            modes_t=modes_t,
            approximation_rank=nystrom_rank,
            bias=True,
            debug=debug,
        )

        self.activation = activation if nonlinear else nn.Identity()

    def set_basis(self, U_approx):
        """接收并传递 Nyström 基"""
        self.sconv.set_basis(U_approx)
        if self.debug:
            print(f"[LiftingOperator] Basis injected: N={self.N}")

    def forward(self, x):
        """
        x: [B, N, T, C_in]
        输出: [B, width, N, T]
        """
        # GraphSpectralConvS_Nystrom 现在会自动处理输入格式
        x = self.sconv(x)  # 自动从 [B, N, T, C] 转为 [B, width, N, T]
        x = self.activation(x)
        return x
