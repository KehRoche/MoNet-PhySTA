import numpy as np
import torch
import torch.nn as nn
from sympy.abc import alpha
import torch.nn.functional as F
import math
import random
from .base import *
from einops import rearrange, repeat
from typing import Optional, Tuple, Sequence,List,Union


class ComplexLinear(nn.Module):
    """
    Complex linear mapping: z_out = W * z_in + b, with W = Wr + i Wi, b = br + i bi.
    Input: real/imag combined as a complex tensor (torch.complex64/128) or as two real tensors.
    We implement using real parameters for Wr, Wi and biases.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_std: Optional[float] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Wr = nn.Parameter(torch.zeros(out_features, in_features))
        self.Wi = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.br = nn.Parameter(torch.zeros(out_features))
            self.bi = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("br", None)
            self.register_parameter("bi", None)
        # initialization
        std = init_std if init_std is not None else 1.0 / math.sqrt(max(1, out_features))
        nn.init.normal_(self.Wr, mean=0.0, std=std)
        nn.init.normal_(self.Wi, mean=0.0, std=std)
        if bias:
            nn.init.zeros_(self.br)
            nn.init.zeros_(self.bi)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: complex tensor shape (..., in_features) with dtype=torch.complex64/128
        returns complex tensor (..., out_features)
        """
        assert torch.is_complex(z), "ComplexLinear expects complex input"
        r = z.real
        i = z.imag
        # compute: (Wr + i Wi) (r + i i) = (Wr r - Wi i) + i (Wr i + Wi r)
        out_r = F.linear(r, self.Wr) - F.linear(i, self.Wi)
        out_i = F.linear(i, self.Wr) + F.linear(r, self.Wi)
        if self.br is not None:
            out_r = out_r + self.br
            out_i = out_i + self.bi
        return torch.complex(out_r, out_i)


def complex_activation_mod(z: torch.Tensor, act: nn.Module) -> torch.Tensor:
    """
    Complex activation by applying a real activation to magnitude and keep phase:
      z -> act(|z|) * exp(i * angle(z))
    This often preserves phase information and is numerically stable.
    """
    mag = torch.abs(z)
    mag_act = act(mag)
    # avoid division by zero for phase: use real/imag via atan2
    angle = torch.atan2(z.imag, z.real)
    real_out = mag_act * torch.cos(angle)
    imag_out = mag_act * torch.sin(angle)
    return torch.complex(real_out, imag_out)


class SpaceTimePositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    a modified sinosoidal PE inspired from the Transformers
    input is (batch, *, nx, ny, t)
    output is (batch, C, nx, ny, t)
    1 comes from the input
    time_exponential_scale comes from the a priori estimate of Navier-Stokes Eqs
    the random feature basis are added with a pointwise conv3d
    """

    def __init__(
        self,
        modes_x,
        modes_t,
        input_shape,
        num_channels,
        spatial_random_feats: bool = False,
        max_time_steps: int = 100,
        time_exponential_scale: float = 1e-2,
        **kwargs,
    ):
        super().__init__()
        assert num_channels % 2 == 0 and num_channels > 3
        self.num_channels = num_channels  # the Euclidean coords
        self.max_time_steps = max_time_steps
        self.time_exponential_scale = time_exponential_scale
        self.modes_x = modes_x
        self.modes_t = modes_t

        self._pe = self._pe_expanded if spatial_random_feats else self._pe
        self._pe(*input_shape)
        if spatial_random_feats:
            in_chan = modes_x * modes_t + 2  # 3 is spatial temporal coords
            self.proj = nn.Conv2d(in_chan, num_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def _pe_expanded(self, *shape):
        nx, ny, nt = shape
        gridx = torch.linspace(0, 1, nx)
        gridy = torch.linspace(0, 1, ny)
        gridt = torch.linspace(0, 1, self.max_time_steps + 1)[1 : nt + 1]
        gridx, gridy, gridt = torch.meshgrid(gridx, gridy, gridt, indexing="ij")
        pe = [gridx, gridy, gridt]

        for i in range(1, self.modes_x + 1):
            basis_x = torch.sin if i % 2 == 0 else torch.cos
            for j in range(1, self.modes_y + 1):
                basis_y = torch.sin if j % 2 == 0 else torch.cos
                for k in range(1, self.modes_t + 1):
                    basis_t = torch.sin if k % 2 == 0 else torch.cos
                    basis = (
                        1
                        / (i * j * k)
                        * torch.exp(self.time_exponential_scale * gridt)
                        * basis_x(torch.pi * i * gridx)
                        * basis_y(torch.pi * j * gridy)
                        * basis_t(torch.pi * k * gridt)
                    )
                    pe.append(basis)
        pe = torch.stack(pe).unsqueeze(0)  # (1, num_channels+3, nx, ny, nt)
        self.pe = pe

    def _pe(self, *shape):
        nx, nt = shape
        gridx = torch.linspace(0, 1, nx)
        gridt = torch.linspace(0, 1, self.max_time_steps + 1)[1 : nt + 1]
        gridx, _gridt = torch.meshgrid(gridx, gridt, indexing="ij")
        pe = [gridx, _gridt]
        for k in range(self.num_channels - 2):
            basis = torch.sin if k % 2 == 0 else torch.cos
            _gridt = torch.exp(self.time_exponential_scale * gridt) * basis(
                torch.pi * (k + 1) * gridt
            )
            _gridt = _gridt.reshape(1, nt).repeat(nx, 1)
            pe.append(_gridt)
        pe = torch.stack(pe).unsqueeze(0)  # (1, num_channels+3, nx, ny, nt)
        self.pe = pe

    def forward(self, v: torch.Tensor):
        if self.pe is None or self.pe.shape[-3:] != v.shape[-3:]:
            *_, nx, nt = v.size()  # (batch, 1, x, y, t)
            self._pe(nx, nt)
        pe = self.pe.to(v.dtype).to(v.device)
        return v + self.proj(pe)


class SpectralFilter(nn.Module):
    """
    Complex spectral filter applied to a frequency-segment.
    Input: x_spec_segment: complex tensor [B, M, T, C_in]
    Output: complex tensor [B, M, T, C_out]
    Uses a learned complex linear mapping + complex activation (magnitude activation by default).
    """
    def __init__(self, C_in: int, C_out: int, complex_act: str = "mod", bias: bool = True):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.linear = ComplexLinear(C_in, C_out, bias=bias)
        self.complex_act_mode = complex_act
        # Real projection fallback (for small models / ablation)
        self.real_proj = nn.Linear(C_out, C_out)  # small real proj after activation (optional)
        self._act = nn.GELU()

    def forward(self, x_spec_segment: torch.Tensor) -> torch.Tensor:
        """
        x_spec_segment: complex tensor [B, M, T, C_in]
        return: complex tensor [B, M, T, C_out]
        """
        assert torch.is_complex(x_spec_segment), "SpectralFilter expects complex-valued spectral input"

        B, M, T, C = x_spec_segment.shape
        # merge leading dims for linear
        x_flat = x_spec_segment.reshape(-1, C)           # (B*M*T, C_in) complex
        z_out = self.linear(x_flat)                      # complex (B*M*T, C_out)

        # complex activation
        if self.complex_act_mode == "mod":
            z_out = complex_activation_mod(z_out, self._act)
        elif self.complex_act_mode == "split":
            # apply nonlinearity separately on real/imag
            r = self._act(z_out.real)
            i = self._act(z_out.imag)
            z_out = torch.complex(r, i)
        elif self.complex_act_mode == "none":
            pass
        else:
            raise ValueError(f"unknown complex_act {self.complex_act_mode}")

        # optional small real projection (applied on real part) to increase expressivity
        # first convert to real pair, then project real part and reconstruct
        r = z_out.real
        i = z_out.imag
        r = self.real_proj(r)
        z_out = torch.complex(r, i)

        out = z_out.view(B, M, T, self.C_out)
        return out

# -------------------------
# Spectral Block: complex linear mapping + skip (mode-wise)
# -------------------------

class SimpleSpectralBlock(nn.Module):
    """
    A single spectral block that maps complex features -> complex features,
    with a residual connection. It acts per (mode,k) and time, but weights are shared.
    Input: z [B, K, T, C]
    """
    def __init__(self, channels: int, hidden: int, activation=nn.GELU()):
        super().__init__()
        self.lin1 = ComplexLinear(channels, hidden)
        self.lin2 = ComplexLinear(hidden, channels)
        self.act = activation

    def forward(self, z: torch.Tensor):
        # z complex [B,K,T,C]
        B, K, T, C = z.shape
        x = self.lin1(z.reshape(-1, C))  # complex
        x = complex_activation_mod(x.view(B, K, T, -1), self.act)
        x = self.lin2(x.reshape(-1, x.shape[-1])).view(B, K, T, C)
        return x + z  # residual




class SpectralBlock(nn.Module):
    """
    SpectralBlock with soft gating + diagnostics.
    """
    def __init__(self,
                 channels: int,
                 hidden: int,
                 activation=nn.GELU(),
                 filters: Optional[List[nn.Module]] = None,
                 gate_hidden: int = 16,
                 ema_momentum: float = 0.9,
                 eps: float = 1e-8,
                 enable_diag: bool = True):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.act = activation
        self.eps = float(eps)
        self.ema_momentum = float(ema_momentum)
        self.enable_diag = enable_diag

        # two-layer complex MLP (residual path)
        self.lin1 = ComplexLinear(channels, hidden)
        self.lin2 = ComplexLinear(hidden, channels)

        # default 4 filters
        if filters is None:
            self.filters = nn.ModuleList([
                SpectralFilter(channels, channels),
                SpectralFilter(channels, channels),
                SpectralFilter(channels, channels),
                SpectralFilter(channels, channels),
            ])
        else:
            assert len(filters) == 4
            self.filters = nn.ModuleList(filters)

        # gate MLP: [lambda_norm, elog_norm] -> 4 logits
        self.gate_net = nn.Sequential(
            nn.Linear(2, gate_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(gate_hidden, 4, bias=True),
        )

    def _ensure_ema_buffer(self, K: int, device, dtype):
        if not hasattr(self, 'ema_energy_buf'):
            buf = torch.zeros(K, dtype=torch.float32, device=device)
            self.register_buffer('ema_energy_buf', buf)
            return True
        else:
            existing = getattr(self, 'ema_energy_buf')
            if existing.numel() != K:
                delattr(self, 'ema_energy_buf')
                buf = torch.zeros(K, dtype=torch.float32, device=device)
                self.register_buffer('ema_energy_buf', buf)
                return True
        return False

    def forward(self, z: torch.Tensor, lambdas: torch.Tensor, baseline: Optional[torch.Tensor] = None):
        """
        z: complex [B, K, T, C]
        lambdas: real [K]
        baseline: optional tensor for spectral error diag (same shape as z)
        """
        if not torch.is_complex(z):
            raise ValueError("SpectralBlock expects complex input")
        if lambdas is None:
            raise ValueError("SpectralBlock requires lambdas")

        B, K, T, C = z.shape
        device = z.device
        diagnostics = {} if self.enable_diag else None

        # 1) per-mode energy
        energy = (z.abs() ** 2).sum(dim=(0, 2, 3)).to(dtype=torch.float32, device=device)  # [K]
        elog = torch.log(energy + self.eps)

        # 2) EMA update
        first_init = self._ensure_ema_buffer(K, device, torch.float32)
        if first_init:
            self.ema_energy_buf.copy_(elog.detach())
        else:
            m = self.ema_momentum
            self.ema_energy_buf.mul_(m).add_((1.0 - m) * elog.detach())
        elog_smooth = self.ema_energy_buf  # [K]

        # 3) norm lambdas & elog
        lambdas = lambdas.to(device=device, dtype=torch.float32)
        lambda_norm = (lambdas - lambdas.mean()) / (lambdas.std(unbiased=False) + 1e-6)
        elog_norm = (elog_smooth - elog_smooth.mean()) / (elog_smooth.std(unbiased=False) + 1e-6)

        # 4) gate -> [K,4]
        gate_in = torch.stack([lambda_norm, elog_norm], dim=-1)  # [K,2]
        gates_logits = self.gate_net(gate_in)
        gates = F.softmax(gates_logits, dim=-1)  # [K,4]

        # ========== Diagnostics ==========
        if diagnostics is not None:
            diagnostics["gates_mean"] = gates.mean(dim=0).detach().cpu()   # [4]
            diagnostics["gates_std"]  = gates.std(dim=0).detach().cpu()    # [4]
            diagnostics["energy_batch"] = energy.detach().cpu()
            diagnostics["elog"] = elog.detach().cpu()
            diagnostics["elog_ema"] = elog_smooth.detach().cpu()

        # 5) apply 4 filters
        outs = [filt(z) for filt in self.filters]  # list of [B,K,T,C]
        w = [gates[:, i].view(1, K, 1, 1) for i in range(4)]
        y = sum(out * wi for out, wi in zip(outs, w))  # [B,K,T,C]

        # 6) complex MLP
        x = self.lin1(y.reshape(-1, C))
        x = complex_activation_mod(x, self.act)
        x = self.lin2(x).view(B, K, T, C)

        # 7) residual
        out = x + z

        # ========== More Diagnostics ==========
        if diagnostics is not None:
            # gradient norm of filters
            grad_norms = []
            for i, filt in enumerate(self.filters):
                if hasattr(filt, "weight") and filt.weight.grad is not None:
                    grad_norms.append(filt.weight.grad.norm().item())
            if grad_norms:
                diagnostics["W_filter_grad_norm_avg"] = sum(grad_norms)/len(grad_norms)

            # spectral reconstruction error
            if baseline is not None:
                err = torch.mean(torch.abs(y - baseline)**2).item()
                diagnostics["spec_L2_err"] = err

        return (out, diagnostics) if self.enable_diag else out



# class SpecGraphFreqNet(nn.Module):
#     def __init__(self,
#                  in_channels, hidden_dim,
#                  energy_splits=(0.8,0.95),
#                  gate_hidden=16):
#         super().__init__()
#         self.F_in      = in_channels
#         self.hidden_dim= hidden_dim
#         self.low_cut, self.mid_cut = energy_splits
#
#         # 四段独立滤波器
#         self.filter_high = SpectralFilter(in_channels, hidden_dim)
#         self.filter_mid  = SpectralFilter(in_channels, hidden_dim)
#         self.filter_low  = SpectralFilter(in_channels, hidden_dim)
#         self.filter_neg  = SpectralFilter(in_channels, hidden_dim)
#
#         # 复合门控网络：基于每个 lambda 值生成四段权重
#         # 输入维度 1 → 隐藏 → 输出 4 → softmax 得到 [w_high, w_mid, w_low, w_neg]
#         self.gate_net = nn.Sequential(
#             nn.Linear(1, gate_hidden, bias=True),
#             nn.ReLU(),
#             nn.Linear(gate_hidden, 4, bias=True),
#         )
#
#         # 最终融合投影回时域特征
#         self.proj = nn.Linear(hidden_dim, in_channels)
#
#     def forward(self, x, eigvecs, lambdas):
#         """
#         x: [B, N, T, F_in]
#         eigvecs: [N, K] or complex
#         lambdas: [K]   频谱索引对应的特征
#         """
#         B, N, T,_ = x.shape
#         device = x.device
#
#         # 1. GFT + FFT → x_spec [B, K, T, F]
#         if torch.is_complex(eigvecs):
#             x_gft = torch.einsum('bntf, nk -> bktf',
#                                  x.to(dtype=torch.complex64),
#                                  eigvecs.conj())
#         else:
#             x_gft = torch.einsum('bntf, nk -> bktf', x, eigvecs)
#         x_spec = torch.fft.fft(x_gft, dim=2)
#
#         # 2. 计算能量并拆分索引（保留用于可视化或对比）
#         energy = (x_spec.abs()**2).sum(dim=(0,2,3))  # [K]
#         neg_mask = lambdas < 0
#         pos_idxs = (~neg_mask).nonzero().squeeze()
#         pos_sorted = pos_idxs[energy[pos_idxs].argsort(descending=True)]
#         k_pos = pos_sorted.numel()
#         cut1, cut2 = int(self.low_cut*k_pos), int(self.mid_cut*k_pos)
#         high_idx = pos_sorted[:cut1]
#         mid_idx  = pos_sorted[cut1:cut2]
#         low_idx  = pos_sorted[cut2:]
#         neg_idx = (lambdas < 0).nonzero(as_tuple=False).flatten()
#
#         # 3. 软门控权重：每个频谱 lambda 都有一个四段权重
#         lam = lambdas.view(-1,1)                            # [K,1]
#         gates = self.gate_net(lam)                          # [K,4]
#         gates = F.softmax(gates, dim=-1)                    # [K,4]
#         w_high, w_mid, w_low, w_neg = gates.unbind(-1)      # 各自 [K]
#
#         # 4. 按段分别应用滤波器，乘以对应 gate 权重后累加
#         # 初始化 y_spec
#         K = x_spec.shape[1]
#         y_spec = torch.zeros([B, K, T, self.hidden_dim], device=x_spec.device, dtype=x_spec.dtype)
#         # 对每一段做变换并加权
#         if len(high_idx)>0:
#             seg = x_spec[:, high_idx]                        # [B, Mh, T, F]
#             out = self.filter_high(seg)                      # [B, Mh, T, hidden_dim]
#             y_spec[:, high_idx] += out * w_high[high_idx].view(1,-1,1,1)
#
#         if len(mid_idx)>0:
#             seg = x_spec[:, mid_idx]
#             out = self.filter_mid(seg)
#             y_spec[:, mid_idx] += out * w_mid[mid_idx].view(1,-1,1,1)
#
#         if len(low_idx)>0:
#             seg = x_spec[:, low_idx]
#             out = self.filter_low(seg)
#             y_spec[:, low_idx] += out * w_low[low_idx].view(1,-1,1,1)
#
#         if len(neg_idx)>0:
#             seg = x_spec[:, neg_idx]
#             out = self.filter_neg(seg)
#             y_spec[:, neg_idx] += out * w_neg[neg_idx].view(1,-1,1,1)
#
#         # 5. IFFT + IGFT → 回到时域
#         y_ifft = torch.fft.ifft(y_spec, dim=2)
#         if not torch.is_complex(eigvecs):
#             y_ifft = y_ifft.real
#
#         y_rec = torch.einsum('bktf, nk -> bntf', y_ifft, eigvecs)
#         # 6. 最后投影到输入维度并残差连接
#         y_out = self.proj(y_rec.real)
#
#         return y_out
#
#

class SpectralConvS(SpectralConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_t: int,
        dim: int = 2,
        bias: bool = False,
        delta: float = 1,
        norm="backward",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=(modes_x, modes_t),
            dim=dim,
            bias=bias,
            norm=norm,
        )

        """
        Spacetime Fourier layer.
        FFT, linear transform, and Inverse FFT.
        focusing on space
        see base.py for the boilerplate
        """
        self.modes_x = modes_x
        self.modes_t = modes_t
        self.delta = delta

    def spectral_conv(self, vh, kx: int, kt: int):
        """
        kx, ky, kt: the number of modes in the input
        matmul the weights with the input
        user defined dimensions
        in the space focused conv
        assert nt <= modes_t not explicitly checked
        """
        bsz = vh.size(0)
        sizes = (bsz, self.out_channels, kx, kt)
        out = torch.zeros(
            *sizes,
            dtype=vh.dtype,
            device=vh.device,
        )
        slice_x = [slice(0, self.modes_x), slice(-self.modes_x, None)]
        st = slice(0, self.modes_t)
        for ix, sx in enumerate(slice_x):
                out[..., sx, st] = self.complex_matmul(
                    vh[..., sx, st], torch.view_as_complex(self.weight[ix])
                )
                # (b, c_i, x, y, t), (c_i, c_o, x, y, t)  -> (b, c_o, x, y, t)
                if self.bias:
                    _bias = self.bias[ix][None, ...]
                    out[..., sx, st] += self.delta * torch.view_as_complex(_bias)
        return out

    def forward(self, v, **kwargs):
        return super().forward(v, **kwargs)


class GraphSpectralConvS(nn.Module):
    """
    Drop-in replacement for SpectralConvS.spectral_conv interface.
    - vh: complex tensor (B, Cin, K, T)
    - spectral_conv(vh, kx, kt, eigvals=None) -> (B, Cout, K, kt) complex
    Partitioning: top-k modes by energy -> low (per-mode weights).
                  remaining -> neg (eigvals<0) / mid / high (shared weights).
    """
    def __init__(self, in_channels, out_channels, modes_x, modes_t,
                 bias=False, delta=1.0, device=None, dtype=torch.float32):
        super().__init__()
        self.Cin = in_channels
        self.Cout = out_channels
        self.low_k = int(modes_x)
        self.modes_t = int(modes_t)
        self.delta = float(delta)
        self.bias_flag = bool(bias)
        self.device = device
        self.dtype = dtype

        # low: per-mode weights (Cin, Cout, low_k, T, 2) real/imag storage
        self.w_low = nn.Parameter(torch.randn(in_channels, out_channels, max(1, self.low_k), self.modes_t, 2, dtype=dtype, device=device) * 0.02)

        # shared weights for neg / mid / high: (Cin, Cout, T, 2)
        self.w_neg  = nn.Parameter(torch.randn(in_channels, out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.02)
        self.w_mid  = nn.Parameter(torch.randn(in_channels, out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.02)
        self.w_high = nn.Parameter(torch.randn(in_channels, out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.02)

        if bias:
            self.b_low  = nn.Parameter(torch.randn(out_channels, max(1, self.low_k), self.modes_t, 2, dtype=dtype, device=device) * 0.01)
            self.b_neg  = nn.Parameter(torch.randn(out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.01)
            self.b_mid  = nn.Parameter(torch.randn(out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.01)
            self.b_high = nn.Parameter(torch.randn(out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.01)

    @staticmethod
    def _to_complex(realimag):
        return torch.view_as_complex(realimag)

    @staticmethod
    def _cmatmul(v, W):
        # v: (B, Cin, M, T) complex, W: (Cin, Cout, T) complex -> (B, Cout, M, T)
        return torch.einsum('b c m t, c o t -> b o m t', v, W)

    def spectral_conv(self, vh, kx: int, kt: int, eigvals: torch.Tensor = None):
        """
        vh: complex (B, Cin, K, T)
        kx: number of spatial modes (K)
        kt: requested time modes (should be <= self.modes_t)
        eigvals: optional real (K,) to separate negative modes
        """
        assert vh.dtype in (torch.complex64, torch.complex128), "vh must be complex"
        B, Cin, K, T = vh.shape
        Tsel = min(kt, self.modes_t, T)

        # compute per-mode energy (sum over batch, channels, time)
        energy = (vh.abs() ** 2).sum(dim=(0,1,3))  # (K,)

        # sort by energy descending (most energetic first)
        idx_by_energy = torch.argsort(energy, descending=True, stable=True)

        low_k = min(self.low_k, K)
        low_idx = idx_by_energy[:low_k]
        rest_idx = idx_by_energy[low_k:]

        # if eigvals provided, extract neg_idx from rest, else neg_idx empty
        if eigvals is not None:
            eigvals = eigvals.to(vh.device)
            neg_mask = eigvals < 0
            # intersection of rest_idx and neg_mask
            if neg_mask.any():
                neg_idx_all = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)
                # keep only those in rest_idx
                # create set-like mask
                rest_mask = torch.zeros(K, dtype=torch.bool, device=vh.device)
                rest_mask[rest_idx] = True
                neg_idx = neg_idx_all[rest_mask[neg_idx_all]] if neg_idx_all.numel() > 0 else torch.empty(0, dtype=torch.long, device=vh.device)
            else:
                neg_idx = torch.empty(0, dtype=torch.long, device=vh.device)
            # remove neg_idx from rest_idx
            if neg_idx.numel() > 0:
                rest_idx = rest_idx[~torch.isin(rest_idx, neg_idx)]
        else:
            neg_idx = torch.empty(0, dtype=torch.long, device=vh.device)

        # split remaining rest_idx into mid / high by energy (half-half)
        m = rest_idx.numel()
        if m > 0:
            half = m // 2
            mid_idx = rest_idx[:half]
            high_idx = rest_idx[half:]
        else:
            mid_idx = torch.empty(0, dtype=torch.long, device=vh.device)
            high_idx = torch.empty(0, dtype=torch.long, device=vh.device)

        # prepare output
        out = torch.zeros(B, self.Cout, K, Tsel, dtype=vh.dtype, device=vh.device)

        # slice vh for each band (only Tsel)
        def slice_v(idx):
            return vh[..., idx, :Tsel] if idx.numel() > 0 else None

        vh_low  = slice_v(low_idx)
        vh_neg  = slice_v(neg_idx)
        vh_mid  = slice_v(mid_idx)
        vh_high = slice_v(high_idx)

        # complex weights
        W_low_c  = self._to_complex(self.w_low)   # (Cin, Cout, low_k, Tsel_or_T)
        W_neg_c  = self._to_complex(self.w_neg)
        W_mid_c  = self._to_complex(self.w_mid)
        W_high_c = self._to_complex(self.w_high)

        # compute low block (vectorized)
        if low_idx.numel() > 0:
            # W_low_c: (Cin, Cout, low_k, T) -> (low_k, Cin, Cout, T)
            W_low_perm = W_low_c.permute(2, 0, 1, 3).contiguous()
            # vh_low: (B, Cin, low_k, T)
            # einsum: 'b c m t, m c o t -> b o m t'
            out_low = torch.einsum('b c m t, m c o t -> b o m t', vh_low, W_low_perm)
            # place into global indices
            for i_local, gidx in enumerate(low_idx.tolist()):
                out[:, :, gidx, :] = out_low[:, :, i_local, :]

            if self.bias_flag:
                b_low_c = self._to_complex(self.b_low)  # (Cout, low_k, T)
                for i_local, gidx in enumerate(low_idx.tolist()):
                    out[:, :, gidx, :] += self.delta * b_low_c[:, i_local, :][None, :, :]

        # helper to apply shared W to band
        def apply_shared(vband, Wc, bparam=None):
            if vband is None:
                return
            out_band = self._cmatmul(vband, Wc)  # (B, Cout, M, Tsel)
            return out_band

        if neg_idx.numel() > 0:
            out_neg = apply_shared(vh_neg, W_neg_c)
            out[:, :, neg_idx, :] = out_neg
            if self.bias_flag:
                out[:, :, neg_idx, :] += self.delta * self._to_complex(self.b_neg)[None, :, None, :]

        if mid_idx.numel() > 0:
            out_mid = apply_shared(vh_mid, W_mid_c)
            out[:, :, mid_idx, :] = out_mid
            if self.bias_flag:
                out[:, :, mid_idx, :] += self.delta * self._to_complex(self.b_mid)[None, :, None, :]

        if high_idx.numel() > 0:
            out_high = apply_shared(vh_high, W_high_c)
            out[:, :, high_idx, :] = out_high
            if self.bias_flag:
                out[:, :, high_idx, :] += self.delta * self._to_complex(self.b_high)[None, :, None, :]

        return out  # complex (B, Cout, K, Tsel)



class SpectralConvT(SpectralConvS):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_t: int,
        delta: float = 1e-1,
        out_steps: int = None,
        norm: str = "backward",
        bias: bool = True,
        temporal_padding: bool = False,
        postprocess: nn.Module = nn.Identity(),
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            modes_x,
            modes_t,
            delta=delta,
            bias=bias,
        )
        self.out_steps = out_steps
        self.temporal_padding = temporal_padding
        self.postprocess = postprocess

        """
        Spacetime Fourier layer used in SFNO. 
        FFT, linear transform, and Inverse FFT.
        arbitrary temporal steps focusing on time  
        """

    def forward(self, v, out_steps: int = None, eigvecs: Optional[torch.Tensor] = None):
        """
        when temporal padding is applied
        the outsteps must be given
        not checked explicitly
        """
        nt = v.size(-1)
        if self.temporal_padding:
            # this is for out conv
            t_pad = v.size(-1)
            v = F.pad(v, (t_pad, 0))
        else:
            t_pad = 0

        if eigvecs is not None:
            eig = eigvecs.to(v.device)  # 不变

            # (B, N, T, C) 用于 GFT
            x_node = v.permute(0, 2, 3, 1)  # inplace 复用 v, 避免再建变量

            # GFT: project node->mode
            if torch.is_complex(eig):
                x_gft = torch.einsum('bntc,nk->bktc', x_node.to(torch.complex64), eig.conj())
            else:
                x_gft = torch.einsum('bntc,nk->bktc', x_node, eig)

            # Time FFT (B, K, T, C) → (B, C, K, T)
            x_spec = torch.fft.fft(x_gft, dim=2).permute(0, 3, 1, 2).contiguous()

            # Spectral conv + postprocess
            v_hat = self.spectral_conv(x_spec, x_spec.size(2), x_spec.size(3))
            v_hat = self.postprocess(v_hat)

            # IFFT: (B, C_out, K, T) → (B, K, T, C_out)
            y_time = torch.fft.ifft(
                v_hat.permute(0, 2, 3, 1).contiguous(),
                n=(out_steps + t_pad) if out_steps is not None else nt,
                dim=2
            )

            # Inverse GFT: (B, K, T, C_out) → (B, N, T, C_out)
            if torch.is_complex(eig):
                y_back = torch.einsum('bktc,nk->bntc', y_time, eig).real
            else:
                y_back = torch.einsum('bktc,nk->bntc', y_time, eig)
                if torch.is_complex(y_back):
                    y_back = y_back.real

            # Output: (B, C_out, N, T)
            out = y_back.permute(0, 3, 1, 2).contiguous()

            # Padding / slicing
            if self.temporal_padding:
                out = out[..., -out_steps:]
            elif out_steps is not None:
                out = out[..., :out_steps]

            return out

        *_, nx, ntp = v.size()  # (b, c, nx, ny, nt)
        v_hat = self.fft(v)
        v_hat = self.spectral_conv(v_hat, nx, ntp // 2 + 1)

        if out_steps is None and self.out_steps is not None:
            out_steps = self.out_steps  # latent_steps
        v_hat = self.postprocess(v_hat)

        v = self.ifft(v_hat, s=(nx, out_steps + t_pad))
        if self.temporal_padding:
            v = v[..., -out_steps:]
        return v

class GraphSpectralConvT(GraphSpectralConvS):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_t: int,
        delta: float = 1e-1,
        out_steps: int = None,
        norm: str = "backward",
        bias: bool = True,
        temporal_padding: bool = False,
        postprocess: nn.Module = nn.Identity(),
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            modes_x,
            modes_t,
            delta=delta,
            bias=bias,
        )
        self.out_steps = out_steps
        self.temporal_padding = temporal_padding
        self.postprocess = postprocess

        """
        Spacetime Fourier layer used in SFNO. 
        FFT, linear transform, and Inverse FFT.
        arbitrary temporal steps focusing on time  
        """

    def forward(self, v, out_steps: int = None, eigvecs: Optional[torch.Tensor] = None):
        """
        when temporal padding is applied
        the outsteps must be given
        not checked explicitly
        """
        nt = v.size(-1)
        if self.temporal_padding:
            # this is for out conv
            t_pad = v.size(-1)
            v = F.pad(v, (t_pad, 0))
        else:
            t_pad = 0

        # --- branch 1: use provided eigvecs (GFT in space), then FFT in time ---
        if eigvecs is not None:
            eig = eigvecs.to(v.device)  # 不变

            # (B, N, T, C) 用于 GFT
            x_node = v.permute(0, 2, 3, 1)  # inplace 复用 v, 避免再建变量

            # GFT: project node->mode
            if torch.is_complex(eig):
                x_gft = torch.einsum('bntc,nk->bktc', x_node.to(torch.complex64), eig.conj())
            else:
                x_gft = torch.einsum('bntc,nk->bktc', x_node, eig)

            # Time FFT (B, K, T, C) → (B, C, K, T)
            x_spec = torch.fft.fft(x_gft, dim=2).permute(0, 3, 1, 2).contiguous()

            # Spectral conv + postprocess
            v_hat = self.spectral_conv(x_spec, x_spec.size(2), x_spec.size(3))
            v_hat = self.postprocess(v_hat)

            # IFFT: (B, C_out, K, T) → (B, K, T, C_out)
            y_time = torch.fft.ifft(
                v_hat.permute(0, 2, 3, 1).contiguous(),
                n=(out_steps + t_pad) if out_steps is not None else nt,
                dim=2
            )

            # Inverse GFT: (B, K, T, C_out) → (B, N, T, C_out)
            if torch.is_complex(eig):
                y_back = torch.einsum('bktc,nk->bntc', y_time, eig).real
            else:
                y_back = torch.einsum('bktc,nk->bntc', y_time, eig)
                if torch.is_complex(y_back):
                    y_back = y_back.real

            # Output: (B, C_out, N, T)
            out = y_back.permute(0, 3, 1, 2).contiguous()

            # Padding / slicing
            if self.temporal_padding:
                out = out[..., -out_steps:]
            elif out_steps is not None:
                out = out[..., :out_steps]

            return out
        # *_, nx, ntp = v.size()  # (b, c, nx, ny, nt)
        # v_hat = self.fft(v)
        # v_hat = self.spectral_conv(v_hat, nx, ntp // 2 + 1)
        #
        # if out_steps is None and self.out_steps is not None:
        #     out_steps = self.out_steps  # latent_steps
        # v_hat = self.postprocess(v_hat)
        #
        # v = self.ifft(v_hat, s=(nx, out_steps + t_pad))
        # if self.temporal_padding:
        #     v = v[..., -out_steps:]
        # return v


class LiftingOperator(nn.Module):
    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_t: int,
        input_shape,
        latent_steps: int = 12,
        norm: str = "backward",
        activation: ActivationType = "GELU",
        beta: float = 0.1,
        spatial_random_feats: bool = False,
        channel_expansion: int = 4,
        nonlinear: bool = True,
        **kwargs,
    ) -> None:
        """
        the latent steps: n_t at hidden layers
        """
        super().__init__()
        if modes_t % 2 != 0:
            pe_modes_t = modes_t - 1
        else:
            pe_modes_t = modes_t

        self.pe = SpaceTimePositionalEncoding(
            modes_x // 2,
            pe_modes_t // 2,
            input_shape=input_shape,
            num_channels=width,
            time_exponential_scale=beta,
            spatial_random_feats=spatial_random_feats,
        )

        in_channels = self.pe.num_channels
        self.norm = LayerNormnd(in_channels)
        self.proj = nn.Conv2d(in_channels, width, kernel_size=1)

        conv_size = [width, width, modes_x, modes_t]
        self.sconv = SpectralConvT(
            *conv_size,
            out_steps=latent_steps,
            norm=norm,
            bias=False,
        )
        self.latent_steps = latent_steps
        if nonlinear:
            self.activation = getattr(nn, activation)()
            self.mlp = PointwiseFFN(width, width, channel_expansion * width, activation)
        else:
            self.activation = nn.Identity()
            self.mlp = nn.Conv3d(width, width, kernel_size=1)

    def forward(self, v,eigvec):
        """
        input: (b, 1, x, y, t)
        output: (b, H, x, y, t_latent)
        the t_latent should be <= the input time steps
        """
        assert self.latent_steps <= v.size(-1)
        for b in [self.pe, self.norm, self.proj]:
            v = b(v)  # (b, 1, x, y, t_in) -> (b, H, x, y, t_latent)
        w = self.mlp(self.sconv(v,eigvecs=eigvec))  # (b, H, x, y, t_latent)
        #
        v = self.activation(v[..., -1:]+ w)
        return v


class OutConv(nn.Module):
    def __init__(
        self,
        modes_x: int,
        modes_t: int,
        delta: float = 0.1,
        out_dim: int = 1,
        diam: float = 1,
        n_grid: int = 64,
        out_steps: int = None,
        spatial_padding: int = 0,
        temporal_padding: bool = True,
        norm: str = "backward",
        **kwargs,
    ) -> None:
        super().__init__()
        """
        from latent steps to output steps
        diam and n_grid are only needed for Helmholtz decomposition
        """
        self.size = [out_dim, out_dim, modes_x, modes_t]
        if out_dim == 2:
            postprocess = HelmholtzProjection(n_grid=n_grid, diam=diam)
        elif out_dim == 1:
            postprocess = nn.Identity()
        self.conv = SpectralConvT(
            *self.size,
            norm=norm,
            delta=delta,
            out_steps=out_steps,
            bias=True,
            temporal_padding=temporal_padding,
            postprocess=postprocess,
        )
        self.n_grid = n_grid
        self.norm = norm
        self.delta = delta
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding

    def forward(self, v, v_res, out_steps: int, **kwargs):
        """
        input v: (b, d, x, y, t_latent)
        d = out_dim = 1 or 2
        input v_res: (b, x, y, t_in) or (b, 2, x, y, t_in)
        after channel reduction and padding length
        v: (b, x, y, t_latent) or (b, 2, x, y, t_latent)
        v_res input (b, x, y, t_out) if out_steps is None
        """
        v_res = v_res.permute(0,3,1,2)
        v = torch.cat([v_res[..., -1:], v], dim=-1)
        if self.spatial_padding > 0:
            sp = self.spatial_padding
            padding_kws = {"pad": (0, 0, sp, sp), "mode": "constant"}
            v = F.pad(v, **padding_kws)

        v = self.conv(v, out_steps=out_steps + 1)
        # if dim reduction is 2, then this v is postprocessed to be divergence free
        # the squeeze(1) would do nothing in the case of velocity

        if self.spatial_padding > 0:
            v = v[..., sp:-sp, :]

        v = v_res[..., -1:] + v[..., -out_steps:]
        return v.squeeze(1)




# -------------------------
# SpecGraphFreqNet (SFNO-style)
# -------------------------
class GTFNO3d(nn.Module):
    def __init__(
            self,
            modes_x: int,
            modes_y: int,
            modes_t: int,
            width: int,
            out_dim: int = 1,
            beta: float = -1e-2,
            delta: float = 1e-1,
            num_spectral_layers: int = 4,
            fft_norm: str = "backward",
            activation: ActivationType = "ReLU",
            spatial_padding: int = 0,
            temporal_padding: bool = True,
            channel_expansion: int = 4,
            spatial_random_feats: bool = False,
            lift_activation: bool = True,
            latent_steps: int = 10,
            output_steps: int = None,
            debug=False,
            **kwargs,
    ):
        super().__init__(
            num_spectral_layers=num_spectral_layers,
            fft_norm=fft_norm,
            activation=activation,
            spatial_padding=spatial_padding,
            channel_expansion=channel_expansion,
            spatial_random_feats=spatial_random_feats,
            lift_activation=lift_activation,
            debug=debug,
            **kwargs,
        )

        """
        The overall network reimplemented to model (2+1)D spatiotemporal PDEs of 
        a scalar field/vector fields of NSE-like equations.

        Major architectural differences:

        1. New lifting operator
            - new PE: since the treatment of grid is different from FNO official code, which give my autograd trouble, new PE is similar to the one used in the Transformers, the time dimension's PE is according to the NSE. The PE occupies the extra channels.
            - new LayerNorm3d: instead of normalizing the input/output pointwisely when preparing the data like the original FNO did, this makes an input-steps agnostic normalization. Note that the global normalization by mean/std of (n, n, n_t)-shaped tensor in the original FNO3d prevents to predict arbitrary time steps.
            - the channel lifting now works pretty much like the depth-wise conv but uses the globally spectral as FNO does. Since there is no need to treat the time steps as channels now it can accept arbitrary time steps in the input.
        2. new out projection: it maps the latent time steps to a given output time steps using FFT's natural super-resolution.
            - output arbitrary steps.
            - aliasing error handled by zero padding
            - the spectral bias works like a source term in the Fredholm integral operator.
        3. n layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv.

        Hyper-params:
        - mode_x, mode_y, mode_t: the number of Fourier modes in the x, y, t dimensions
        - width: the number of channels in the latent space   
        - num_spectral_layers: the number of spectral conv layers, the first layer is in the lifting operator  
        - spatial_padding: the padding size in the spatial dimensions
        - temporal_padding: whether to pad the temporal dimension, by default it is True, recommended to keep it True to avoid aliasing error
        - out_steps: the number of output time steps, if None, it will be set to the temporal dimension of the input
        - activation: the activation function, users provide string that directly pulls from nn. default: ReLU
        - lift_activation: whether to use activation in the lifting operator
        - spatial_random_feats: whether to use spatial random features in the lifting operator
        - channel_expansion: the number of channels in the MLP, default: 128

        Grid information:
        - diam: the diameter of the domain, only used in the Helmholtz decomposition
        - n_grid: the grid size of the training data, only needed for building the fft mesh for the Helmholtz decompostion, in the forward pass the size is arbitrary (if different from the n_grid, Helmholtz layer will re-build the fft mesh, which introduces a tiny overhead)

        Several key hyper-params that is different from FNO3d:
        - beta: the exponential scaling factor for the time PE, ideally it should match the a priori estimate the energy of the NSE
        - delta: the strength of the final skip-connection.
        - latent steps: the number of time steps in the hidden layers, this is independent of the input/output steps; chosing it >= 3/2 of input length is similar to zero padding of FFT to avoid aliasing due to non-periodic in the temporal dimension
        - dim_reduction: 1 for scalar field such as vorticity, 2 for vector field such as velocity

        input: w(x, y, t) in the shape of (bsz, x, y, t)
        output: w(x, y, t) in the shape of (bsz, x, y, t)
        """

        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_t = modes_t
        self.width = width

        assert num_spectral_layers > 1
        num_spectral_layers -= 1
        # the lifting operator has already an sconv

        self._set_spectral_layers(
            num_spectral_layers,
            [modes_x, modes_y, modes_t],
            width,
            spectral_conv=SpectralConvS,
            mlp=PointwiseFFN,
            linear=nn.Conv3d,
            activation=activation,
            channel_expansion=channel_expansion,
        )

        self.lifting_operator = LiftingOperator(
            width,
            modes_x,
            modes_y,
            modes_t,
            latent_steps=latent_steps,
            norm=fft_norm,
            beta=beta,
            activation=activation,
            spatial_random_feats=spatial_random_feats,
            channel_expansion=channel_expansion,
            nonlinear=lift_activation,
        )

        self.output_operator = OutConv(
            modes_x,
            modes_y,
            modes_t,
            out_dim=out_dim,
            delta=delta,
            out_steps=output_steps,
            spatial_padding=spatial_padding,
            temporal_padding=temporal_padding,
            norm=fft_norm,
        )

        self.reduction = nn.Conv3d(width, 1, kernel_size=1)
        self.out_steps = output_steps
        self.debug = debug

    def forward(self, x: torch.Tensor, eigvecs: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
        """
        x: real [B, N, T, F_in]
        eigvecs: [N, K] or complex
        lambdas: [K]
        returns: real [B, N, T, F_in] (residual added)
        """
        
        if out_steps is None:
            out_steps = self.out_steps if self.out_steps is not None else v.size(-1)
        v_res = v  # save skip connection
        v = rearrange(v, "b x y t -> b 1 x y t")
        v = self.lifting_operator(v)  # [b, 1, x, y, T] -> [b, H, x, y, T]

        for conv, mlp, w, nonlinear in zip(
            self.spectral_conv, self.mlp, self.w, self.activations
        ):
            x1 = conv(v)  # (b,H,x,y,t)
            x1 = mlp(x1)  # conv3d (b, H, x, y, t) -> (b, H, x, y, t)
            x2 = w(v)
            v = x1 + x2
            v = nonlinear(v)

        v = self.reduction(v)  # (b, H, x, y, t) -> (b, 1, x, y, t)
        v = self.output_operator(
            v, v_res, out_steps=out_steps
        )  # (b,1,x,y,t) -> (b,x,y,t)
        return v
        
        
        B, N, T, C = x.shape
        device = x.device
        dtype = x.dtype

        # 1) GFT (project spatially). If eigvecs complex, use conjugate.
        if torch.is_complex(eigvecs):
            x_c = x.to(dtype=torch.complex64)
            x_gft = torch.einsum('bntc,nk->bktc', x_c, eigvecs.conj().to(device))
        else:
            x_gft = torch.einsum('bntc,nk->bktc', x.to(device), eigvecs.to(device))

        # 2) FFT in time dimension -> complex spectral coeffs [B, K, T, C]
        x_spec = torch.fft.fft(x_gft, dim=2)

        # 3) Lifting operator -> complex latent [B, K, T, H]
        z = self.lifting(x_spec)  # complex

        # 4) Spectral blocks (operate in mode domain, time preserved)
        for block in self.spec_blocks:
            z = block(z)
        # 5) Project complex latent to real channel space (take real part after ifft)
        # first reshape and convert to complex->real projection
        Bk, K, Tt, H = z.shape
        # inverse FFT in time domain
        y_ifft = torch.fft.ifft(z, dim=2)  # complex [B,K,T,H]
        # inverse GFT: project back to node domain
        if torch.is_complex(eigvecs):
            y_back = torch.einsum('bktc,nk->bntc', y_ifft, eigvecs.to(device))
            y_back_real = y_back.real
        else:
            y_back = torch.einsum('bktc,nk->bntc', y_ifft, eigvecs.to(device))
            # y_back may be complex (if z complex) - take real part
            y_back_real = y_back.real if torch.is_complex(y_back) else y_back

        # optional postprocess in node domain (e.g., Helmholtz)
        if self.postprocess is not None:
            y_back_real = self.postprocess(y_back_real)

        # final pointwise projection and residual
        y_proj = self.final_complex_to_real(y_back_real.reshape(-1, H)).reshape(B, N, Tt, C)
        out = x + y_proj  # residual

        return out.float()


class GTFNO2d(FNOBase):
    def __init__(
            self,
            x: int,
            t: int,
            width: int,
            out_dim: int = 1,
            beta: float = -1e-2,
            delta: float = 1e-1,
            num_spectral_layers: int = 2,
            fft_norm: str = "backward",
            activation: ActivationType = "ReLU",
            spatial_padding: int = 0,
            temporal_padding: bool = True,
            channel_expansion: int = 4,
            spatial_random_feats: bool = False,
            lift_activation: bool = True,
            latent_steps: int = 12,
            output_steps: int = 12,
            debug=False,
            **kwargs,
    ):
        super().__init__(
            num_spectral_layers=num_spectral_layers,
            fft_norm=fft_norm,
            activation=activation,
            spatial_padding=spatial_padding,
            channel_expansion=channel_expansion,
            spatial_random_feats=spatial_random_feats,
            lift_activation=lift_activation,
            debug=debug,
            **kwargs,
        )

        modes_x = x//4
        modes_t = t//2
        self.width = width

        assert num_spectral_layers > 1
        num_spectral_layers -= 1
        # the lifting operator has already an sconv

        self.spectral_conv = nn.ModuleList(
            [
                # GraphSpectralConvT(width, width, modes_x, modes_t,out_steps=t)
                SpectralConvT(width, width, modes_x, modes_t,out_steps=t)
                for _ in range(num_spectral_layers)
            ]
        )

        self.mlp = nn.ModuleList(
            [MLP(width, width, width) for _ in range(num_spectral_layers)]
        )

        self.w = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(num_spectral_layers)]
        )

        self.activations = nn.ModuleList(
            [nn.GELU() for _ in range(num_spectral_layers)]
        )


        self.lifting_operator = LiftingOperator(
            width,
            modes_x,
            modes_t,
            input_shape=(x,t),
            latent_steps=latent_steps,
            norm=fft_norm,
            beta=beta,
            activation=activation,
            spatial_random_feats=spatial_random_feats,
            channel_expansion=channel_expansion,
            nonlinear=lift_activation,
        )

        self.output_operator = OutConv(
            modes_x,
            modes_t,
            out_dim=out_dim,
            delta=delta,
            out_steps=output_steps,
            spatial_padding=spatial_padding,
            temporal_padding=temporal_padding,
            norm=fft_norm,
        )

        self.reduction = nn.Conv2d(width, 1, kernel_size=1)
        self.out_steps = output_steps
        self.debug = debug

    def forward(self, v: torch.Tensor, eigvecs: torch.Tensor, lambdas: torch.Tensor,out_steps=None) -> torch.Tensor:
        """
        x: real [B, N, T, F_in]
        eigvecs: [N, K] or complex
        lambdas: [K]
        returns: real [B, N, T, F_in] (residual added)
        """

        if out_steps is None:
            out_steps = self.out_steps if self.out_steps is not None else v.size(-1)
        v_res = v  # save skip connection
        v = rearrange(v, "b x t f -> b f x t")
        v = self.lifting_operator(v,eigvecs)  # [b, f, x, T] -> [b, H, x, T]

        for conv, mlp, w, nonlinear in zip(
                self.spectral_conv, self.mlp, self.w, self.activations):
            x1 = conv(v,eigvecs=eigvecs)  # (b,H,x,y,t)
            x1 = mlp(x1)  # conv3d (b, H, x, y, t) -> (b, H, x, y, t)
            x2 = w(v)
            v = x1 + x2
            v = nonlinear(v)

        v = self.reduction(v)  # (b, c, x,t) -> (b, 1, x, t)
        # v = self.output_operator(
        #     v, v_res, out_steps=out_steps
        # )  # (b,1,x,t) -> (b,x,t)
        return v.permute(0,2,3,1)
        #return v.unsqueeze(-1)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, activation=True):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x):
        for layer in [self.mlp1, self.activation, self.mlp2]:
            x = layer(x)
        return x