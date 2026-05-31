import numpy as np
import torch
import torch.nn as nn
from sympy.abc import alpha
from einops import rearrange
from typing import List


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
                y_back = torch.einsum('bktc,nk->bntc', y_time.real, eig)
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


#--------------TGSSP-----------------------
# class GraphSpectralConvS(nn.Module):
#     """
#     Modified: low_k keeps per-mode independent weights (mode_n).
#     mid/high each have:
#       - a shared kernel (w_mid / w_high)
#       - a per-mode non-negative alpha vector, stored only for mid_k / high_k (fixed sizes)
#     forward 兼容老签名，但真正的运算在 spectral_conv 里。
#     """
#     def __init__(self, in_channels, out_channels, N, modes_t,
#                  energy_splits,
#                  time_feat_dim: int = 8, bias=False, device=None, dtype=torch.float32):
#         super().__init__()
#         self.Cin = in_channels
#         self.Cout = out_channels
#         self.K = int(N)
#         self.modes_t = int(modes_t)
#         self.device = device
#         self.dtype = dtype
# 
#         s0, s1 = float(energy_splits[0]), float(energy_splits[1])
#         # low_k is at least 1 if K>=1.
#         low_k = max(1, int(round(s0 * self.K)))
#         # middle part count from fractional interval (s1 - s0)
#         mid_k = int(round((s1 - s0) * self.K))
#         # remaining goes to high
#         high_k = max(0, self.K - low_k - mid_k)
# 
#         # store as attributes for later use
#         self.low_k = int(low_k)
#         self.mid_k = int(mid_k)
#         self.high_k = int(high_k)
# 
# 
#         # --- low per-mode independent slots (size = low_k) ---
#         self.w_low = nn.Parameter(
#             torch.randn(out_channels, out_channels, max(1, self.low_k), self.modes_t, 2,
#                         dtype=dtype, device=device) * 0.02
#         )
#         # --- shared kernels for neg / mid / high (each independent) ---
#         self.w_neg = nn.Parameter(
#             torch.randn(out_channels, out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.02
#         )
#         self.w_mid = nn.Parameter(
#             torch.randn(out_channels, out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.02
#         )
#         self.w_high = nn.Parameter(
#             torch.randn(out_channels, out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.02
#         )
# 
# 
#         # --- store per-band alphas only for mid_k / high_k (non-negative via softplus) ---
#         if self.mid_k > 0:
#             init_mid = 0.1
#             alpha_mid_raw = torch.full((self.mid_k,), init_mid, dtype=dtype, device=device)
#             self.alpha_mid_raw = nn.Parameter(alpha_mid_raw)
#         else:
#             self.alpha_mid_raw = None
#         if self.high_k > 0:
#             init_high = 0.1
#             alpha_high_raw = torch.full((self.high_k,), init_high, dtype=dtype, device=device)
#             self.alpha_high_raw = nn.Parameter(alpha_high_raw)
#         else:
#             self.alpha_high_raw = None
# 
#         if bias:
#             self.b_modes = nn.Parameter(torch.zeros(out_channels, self.K, self.modes_t, 2, dtype=dtype, device=device))
#         else:
#             self.b_modes = None
# 
#         # gate MLP: outputs 4 values per time step (neg, low, mid, high)
#         self.gate_mlp = nn.Sequential(
#             nn.Linear(self.Cin*2, 16),
#             nn.ReLU(),
#             nn.Linear(16, self.Cout)
#         )
# 
#     @staticmethod
#     def _to_complex(realimag):
#         return torch.view_as_complex(realimag)
# 
#     # === 核心频域卷积（供 S/T 共用） ===
#     def spectral_conv(self, vh: torch.Tensor, eigvals: torch.Tensor, time_feats: Optional[torch.Tensor] = None):
#         """
#         vh: (B, Cin, K, T) complex (already in spectral domain)
#         eigvals: (K,) real, to split bands
#         time_feats: (B, T, F_time) or None
#         return: (B, Cout, K, Tsel) complex
#         """
#         B, Cin, K_in, T = vh.shape
#         assert K_in == self.K, f"vh K={K_in} must match init K={self.K}"
#         device = vh.device
#         Tsel = min(T, self.modes_t)
# 
#         eigvals = eigvals.to(device)
#         neg_idx = torch.nonzero(eigvals < 0, as_tuple=False).squeeze(-1)
#         pos_idx = torch.nonzero(eigvals >= 0, as_tuple=False).squeeze(-1)
# 
#         # positive bands split: low / mid / high
#         n_pos = pos_idx.numel()
#         n_low = min(self.low_k, n_pos)
#         rem = max(0, n_pos - n_low)
#         desired_mid = min(self.mid_k, rem)
#         desired_high = rem - desired_mid
# 
#         def clamp(idx):
#             if idx.numel() == 0:
#                 return idx
#             return idx[(idx >= 0) & (idx < self.K)]
# 
#         low_idx  = clamp(pos_idx[:n_low]) if n_low > 0 else torch.empty(0, dtype=torch.long, device=device)
#         mid_idx  = clamp(pos_idx[n_low:n_low + desired_mid]) if desired_mid > 0 else torch.empty(0, dtype=torch.long, device=device)
#         high_idx = clamp(pos_idx[n_low + desired_mid:n_low + desired_high + desired_mid]) if desired_high > 0 else torch.empty(0, dtype=torch.long, device=device)
#         neg_idx  = clamp(neg_idx)
# 
#         out = torch.zeros(B, self.Cout, self.K, Tsel, dtype=vh.dtype, device=device)
# 
#         W_low_c  = self._to_complex(self.w_low[..., :Tsel])   # (Cin, Cout, low_k, Tsel)
#         W_neg_c  = self._to_complex(self.w_neg[..., :Tsel])   # (Cin, Cout, Tsel)
#         W_mid_c  = self._to_complex(self.w_mid[..., :Tsel])   # (Cin, Cout, Tsel)
#         W_high_c = self._to_complex(self.w_high[..., :Tsel])  # (Cin, Cout, Tsel)
#         b_c = self._to_complex(self.b_modes[..., :Tsel]) if self.b_modes is not None else None
# 
#         # NEG band
#         if neg_idx.numel() > 0:
#             vh_neg = vh[:, :, neg_idx, :Tsel]
#             out_neg = torch.einsum('b c m t, c o t -> b o m t', vh_neg, W_neg_c)
#             out[:, :, neg_idx, :] = out_neg
#             if b_c is not None:
#                 out[:, :, neg_idx, :] += b_c[None, :, neg_idx, :]
# 
#         # LOW band
#         if low_idx.numel() > 0:
#             vh_low = vh[:, :, low_idx, :Tsel]
#             m_low = low_idx.numel()
#             Wl_sel = W_low_c[..., :m_low, :]
#             Wl_perm = Wl_sel.permute(2,0,1,3).contiguous()
#             out_low = torch.einsum('b c m t, m c o t -> b o m t', vh_low, Wl_perm)
#             out[:, :, low_idx, :] = out_low
#             if b_c is not None:
#                 out[:, :, low_idx, :] += b_c[None, :, low_idx, :]
# 
#         # MID band
#         if mid_idx.numel() > 0:
#             vh_mid = vh[:, :, mid_idx, :Tsel]
#             out_mid = torch.einsum('b c m t, c o t -> b o m t', vh_mid, W_mid_c)
#             if self.alpha_mid_raw is not None:
#                 alpha_mid = F.softplus(self.alpha_mid_raw)
#                 a_mid = alpha_mid[: mid_idx.numel()].view(1, 1, -1, 1)
#             else:
#                 a_mid = torch.ones(1, 1, mid_idx.numel(), 1, device=device, dtype=vh.real.dtype)
#             out[:, :, mid_idx, :] = out_mid * a_mid
#             if b_c is not None:
#                 out[:, :, mid_idx, :] += b_c[None, :, mid_idx, :]
# 
#         # HIGH band
#         if high_idx.numel() > 0:
#             vh_high = vh[:, :, high_idx, :Tsel]
#             out_high = torch.einsum('b c m t, c o t -> b o m t', vh_high, W_high_c)
#             if self.alpha_high_raw is not None:
#                 alpha_high = F.softplus(self.alpha_high_raw)
#                 a_high = alpha_high[: high_idx.numel()].view(1, 1, -1, 1)
#             else:
#                 a_high = torch.ones(1, 1, high_idx.numel(), 1, device=device, dtype=vh.real.dtype)
#             out[:, :, high_idx, :] = out_high * a_high
#             if b_c is not None:
#                 out[:, :, high_idx, :] += b_c[None, :, high_idx, :]
# 
#         # time gate（若没给 time_feats，则退化为 1）
#         if time_feats is not None:
#             T_in = time_feats.size(1)
#             t_use = min(Tsel, T_in)
#             #B,T,C
#             time_feats = time_feats[:, :, :t_use].transpose(1, 2)
#             time_feats_reim = torch.cat([time_feats.real, time_feats.imag], dim=-1)  # (B, 2C, T)
#             gate_raw = self.gate_mlp(time_feats_reim)
#             gate = torch.sigmoid(gate_raw).permute(0, 2, 1).unsqueeze(2)  # (B,4,1,t_use)
#             # 若 Tsel > t_use，用最后一个时间步的门控填充
#             if t_use < Tsel:
#                 last = gate[..., -1:].expand(-1, -1, -1, Tsel - t_use)
#                 gate = torch.cat([gate, last], dim=-1)
#         else:
#             gate = torch.ones(vh.size(0), 4, 1, Tsel, device=device, dtype=vh.real.dtype)
# 
#         out_final = torch.zeros_like(out)
#         def apply_gate(idx, chan):
#             if idx.numel() == 0:
#                 return
#             m = idx.numel()
#             g = gate[:, chan:chan+1, :, :].expand(-1, 1, m, -1)  # (B,1,m,Tsel)
#             out_final[:, :, idx, :] = g * out[:, :, idx, :]
#
# 
# 
# 
#         # 0/1/2/3 -> neg/low/mid/high
#         apply_gate(neg_idx, 0)
#         apply_gate(low_idx, 1)
#         apply_gate(mid_idx, 2)
#         apply_gate(high_idx, 3)
#         return out_final
#     # 兼容老调用：把参数转给 spectral_conv
#     def forward(self, vh, eigvecs=None, eigvals=None, time_feats=None):
#         assert eigvals is not None, "eigvals must be provided"
#         return self.spectral_conv(vh, eigvals, time_feats)
# 
# 


from typing import Optional

class GraphSpectralConvS(nn.Module):
    def __init__(self, in_channels, out_channels, N, modes_t,
                 energy_splits,
                 time_feat_dim: int = 8, bias=False, device=None, dtype=torch.float32):
        super().__init__()
        self.Cin = in_channels
        self.Cout = out_channels
        self.K = int(N)
        self.modes_t = int(modes_t)
        self.device = device
        self.dtype = dtype
        self.time_feat_dim = int(time_feat_dim)

        s0, s1 = float(energy_splits[0]), float(energy_splits[1])
        low_k = max(1, int(round(s0 * self.K)))
        mid_k = int(round((s1 - s0) * self.K))
        high_k = max(0, self.K - low_k - mid_k)
        self.low_k = int(low_k)
        self.mid_k = int(mid_k)
        self.high_k = int(high_k)

        # 权重
        self.w_low = nn.Parameter(
            torch.randn(out_channels, out_channels, max(1, self.low_k), self.modes_t, 2,
                        dtype=dtype, device=device) * 0.02
        )
        self.w_neg = nn.Parameter(
            torch.randn(out_channels, out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.02
        )
        self.w_mid = nn.Parameter(
            torch.randn(out_channels, out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.02
        )
        self.w_high = nn.Parameter(
            torch.randn(out_channels, out_channels, self.modes_t, 2, dtype=dtype, device=device) * 0.02
        )

        if self.mid_k > 0:
            self.alpha_mid_raw = nn.Parameter(torch.full((self.mid_k,), 0.1, dtype=dtype, device=device))
        else:
            self.alpha_mid_raw = None
        if self.high_k > 0:
            self.alpha_high_raw = nn.Parameter(torch.full((self.high_k,), 0.1, dtype=dtype, device=device))
        else:
            self.alpha_high_raw = None

        if bias:
            self.b_modes = nn.Parameter(torch.zeros(out_channels, self.K, self.modes_t, 2, dtype=dtype, device=device))
        else:
            self.b_modes = None

        # 门控 MLP：每时间步 -> 4 个频带（neg/low/mid/high）
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.time_feat_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    @staticmethod
    def _to_complex(realimag):
        return torch.view_as_complex(realimag)

    # === 核心频域卷积（供 S/T 共用） ===
    def spectral_conv(self, vh: torch.Tensor, eigvals: torch.Tensor, time_feats: Optional[torch.Tensor] = None):
        B, Cin, K_in, T = vh.shape
        assert K_in == self.K, f"vh K={K_in} must match init K={self.K}"
        device = vh.device
        Tsel = min(T, self.modes_t)
        eigvals = eigvals.to(device)

        # split indices
        neg_idx = torch.nonzero(eigvals < 0, as_tuple=False).squeeze(-1)
        pos_idx = torch.nonzero(eigvals >= 0, as_tuple=False).squeeze(-1)

        n_pos = pos_idx.numel()
        n_low = min(self.low_k, n_pos)
        rem = max(0, n_pos - n_low)
        desired_mid = min(self.mid_k, rem)
        desired_high = rem - desired_mid

        def clamp(idx):
            if idx.numel() == 0:
                return idx
            return idx[(idx >= 0) & (idx < self.K)]

        low_idx  = clamp(pos_idx[:n_low]) if n_low > 0 else torch.empty(0, dtype=torch.long, device=device)
        mid_idx  = clamp(pos_idx[n_low:n_low + desired_mid]) if desired_mid > 0 else torch.empty(0, dtype=torch.long, device=device)
        high_idx = clamp(pos_idx[n_low + desired_mid:n_low + desired_high + desired_mid]) if desired_high > 0 else torch.empty(0, dtype=torch.long, device=device)
        neg_idx  = clamp(neg_idx)

        out = torch.zeros(B, self.Cout, self.K, Tsel, dtype=vh.dtype, device=device)
        W_low_c  = self._to_complex(self.w_low[..., :Tsel])
        W_neg_c  = self._to_complex(self.w_neg[..., :Tsel])
        W_mid_c  = self._to_complex(self.w_mid[..., :Tsel])
        W_high_c = self._to_complex(self.w_high[..., :Tsel])
        b_c = self._to_complex(self.b_modes[..., :Tsel]) if self.b_modes is not None else None

        # NEG
        if neg_idx.numel() > 0:
            vh_neg = vh[:, :, neg_idx, :Tsel]
            out_neg = torch.einsum('b c m t, c o t -> b o m t', vh_neg, W_neg_c)
            out[:, :, neg_idx, :] = out_neg
            if b_c is not None:
                out[:, :, neg_idx, :] += b_c[None, :, neg_idx, :]

        # LOW
        if low_idx.numel() > 0:
            vh_low = vh[:, :, low_idx, :Tsel]
            m_low = low_idx.numel()
            Wl_sel = W_low_c[..., :m_low, :]
            Wl_perm = Wl_sel.permute(2, 0, 1, 3).contiguous()  # (m, Cout, Cout, Tsel)
            out_low = torch.einsum('b c m t, m c o t -> b o m t', vh_low, Wl_perm)
            out[:, :, low_idx, :] = out_low
            if b_c is not None:
                out[:, :, low_idx, :] += b_c[None, :, low_idx, :]

        # MID
        if mid_idx.numel() > 0:
            vh_mid = vh[:, :, mid_idx, :Tsel]
            out_mid = torch.einsum('b c m t, c o t -> b o m t', vh_mid, W_mid_c)
            if self.alpha_mid_raw is not None:
                alpha_mid = F.softplus(self.alpha_mid_raw)
                a_mid = alpha_mid[: mid_idx.numel()].view(1, 1, -1, 1)
            else:
                a_mid = torch.ones(1, 1, mid_idx.numel(), 1, device=device, dtype=vh.real.dtype)
            out[:, :, mid_idx, :] = out_mid * a_mid
            if b_c is not None:
                out[:, :, mid_idx, :] += b_c[None, :, mid_idx, :]

        # HIGH
        if high_idx.numel() > 0:
            vh_high = vh[:, :, high_idx, :Tsel]
            out_high = torch.einsum('b c m t, c o t -> b o m t', vh_high, W_high_c)
            if self.alpha_high_raw is not None:
                alpha_high = F.softplus(self.alpha_high_raw)
                a_high = alpha_high[: high_idx.numel()].view(1, 1, -1, 1)
            else:
                a_high = torch.ones(1, 1, high_idx.numel(), 1, device=device, dtype=vh.real.dtype)
            out[:, :, high_idx, :] = out_high * a_high
            if b_c is not None:
                out[:, :, high_idx, :] += b_c[None, :, high_idx, :]

        # time gate（若没给 time_feats，则退化为 1）
        if time_feats is not None:
            T_in = time_feats.size(1)
            t_use = min(Tsel, T_in)
            tf = time_feats[:, :t_use]
            if torch.is_complex(tf):
                tf_reim = torch.cat([tf.real, tf.imag], dim=-1)  # (B, t_use, 2F)
            else:
                tf_reim = torch.cat([tf, tf], dim=-1)            # 若为实数，则复制一份以满足 2F
            F2_expected = self.time_feat_dim * 2
            F2_in = tf_reim.size(-1)
            if F2_in >= F2_expected:
                tf_reim = tf_reim[..., :F2_expected]
            else:
                pad = F2_expected - F2_in
                tf_reim = F.pad(tf_reim, (0, pad), mode="constant", value=0.0)

            gate_raw = self.gate_mlp(tf_reim.reshape(-1, F2_expected))  # (B*t_use, 4)
            gate = torch.sigmoid(gate_raw).view(B, t_use, 4).permute(0, 2, 1).unsqueeze(2)  # (B,4,1,t_use)
            if t_use < Tsel:
                last = gate[..., -1:].expand(-1, -1, -1, Tsel - t_use)
                gate = torch.cat([gate, last], dim=-1)
        else:
            gate = torch.ones(vh.size(0), 4, 1, Tsel, device=device, dtype=vh.real.dtype)
            tf_reim = None

        out_final = torch.zeros_like(out)

        def apply_gate(idx, chan):
            if idx.numel() == 0:
                return
            m = idx.numel()
            g = gate[:, chan:chan+1, :, :].expand(-1, 1, m, -1)  # (B,1,m,Tsel)
            out_final[:, :, idx, :] = g * out[:, :, idx, :]

        apply_gate(neg_idx, 0)
        apply_gate(low_idx, 1)
        apply_gate(mid_idx, 2)
        apply_gate(high_idx, 3)

        return out_final

    def forward(self, vh, eigvecs=None, eigvals=None, time_feats=None):
        assert eigvals is not None, "eigvals must be provided"
        return self.spectral_conv(vh, eigvals, time_feats)

class GraphSpectralConvT(GraphSpectralConvS):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        N: int,
        modes_t: int,
        energy_splits: List[float],
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
            N,
            modes_t,
            energy_splits,
            bias=bias,
        )
        self.out_steps = out_steps
        self.temporal_padding = temporal_padding
        self.postprocess = postprocess

    def forward(self, v, out_steps: int = None,
                eigvecs: Optional[torch.Tensor] = None,
                eigvals: Optional[torch.Tensor] = None,
                time_feats: Optional[torch.Tensor] = None):
        """
        v: (B, C_in, N, T)
        eigvecs: (N, K) 复或实
        eigvals: (K,)
        time_feats: (B, T, F_time) or None
        """
        assert eigvecs is not None, "eigvecs must be provided"
        nt = v.size(-1)
        if self.temporal_padding:
            t_pad = v.size(-1)
            v = F.pad(v, (t_pad, 0))
        else:
            t_pad = 0

        eig = eigvecs.to(v.device)
        # node -> mode (GFT)
        x_node = v.permute(0, 2, 3, 1)  # (B, N, T, C)
        if torch.is_complex(eig):
            x_gft = torch.einsum('bntc,nk->bktc', x_node.to(torch.complex64), eig.conj())
        else:
            x_gft = torch.einsum('bntc,nk->bktc', x_node, eig)

        # FFT in time; to (B, C, K, T)
        x_spec = torch.fft.fft(x_gft, dim=2).permute(0, 3, 1, 2).contiguous()
        time_feats_frq = torch.fft.fft(time_feats, dim=2).contiguous()

        # 频域卷积（修复点：这里调用父类的 spectral_conv）
        v_hat = self.spectral_conv(x_spec, eigvals=eigvals, time_feats=time_feats_frq)
        v_hat = self.postprocess(v_hat)

        # IFFT back (B, K, T, C_out)
        y_time = torch.fft.ifft(
            v_hat.permute(0, 2, 3, 1).contiguous(),
            n=(out_steps + t_pad) if out_steps is not None else nt,
            dim=2
        )

        # inverse GFT -> (B, N, T, C_out)
        if torch.is_complex(eig):
            y_back = torch.einsum('bktc,nk->bntc', y_time, eig).real
        else:
            y_back = torch.einsum('bktc,nk->bntc', y_time.real, eig)
            if torch.is_complex(y_back):
                y_back = y_back.real

        out = y_back.permute(0, 3, 1, 2).contiguous()
        if self.temporal_padding:
            out = out[..., -out_steps:]
        elif out_steps is not None:
            out = out[..., :out_steps]
        return out

class LiftingOperator(nn.Module):
    def __init__(
        self,
        input:int,
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
            num_channels=input,
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

class GTFNO2d(FNOBase):
    def __init__(
            self,
            N: int,
            T: int,
            input: int,
            width: int,
            energy_splits:List[float],
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
        modes_t = T//2
        self.width = width

        self.lifting_operator = LiftingOperator(
            input,
            width,
            N//4,
            modes_t,
            input_shape=(N,T),
            latent_steps=latent_steps,
            norm=fft_norm,
            beta=beta,
            activation=activation,
            spatial_random_feats=spatial_random_feats,
            channel_expansion=channel_expansion,
            nonlinear=lift_activation,
        )

        assert num_spectral_layers > 1
        num_spectral_layers -= 1
        # the lifting operator has already an sconv

        self.spectral_conv = nn.ModuleList(
            [
                GraphSpectralConvT(input, width, N, modes_t,energy_splits,out_steps=T)
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

        self.reduction = nn.Conv2d(width, 1, kernel_size=1)
        self.out_steps = output_steps
        self.debug = debug

    def forward(self, v: torch.Tensor, eigvecs: torch.Tensor, eigval: torch.Tensor,time_feats: torch.Tensor,out_steps=None) -> torch.Tensor:
        """
        x: real [B, N, T, F_in]
        eigvecs: [N, K] or complex
        lambdas: [K]
        returns: real [B, N, T, F_in] (residual added)
        """

        if out_steps is None:
            out_steps = self.out_steps if self.out_steps is not None else v.size(-1)
        v = rearrange(v, "b x t f -> b f x t")
        v = self.lifting_operator(v,eigvecs)  # [b, f, x, T] -> [b, width, x, T]

        for conv, mlp, w, nonlinear in zip(
                self.spectral_conv, self.mlp, self.w, self.activations):
            x1 = conv(v,eigvecs=eigvecs,eigvals=eigval,time_feats=time_feats)  # (b,H,x,y,t)
            x1 = mlp(x1)  # conv3d (b, H, x, y, t) -> (b, H, x, y, t)
            x2 = w(v)
            v = x1 + x2
            v = nonlinear(v)

        v = self.reduction(v)  # (b, c, x,t) -> (b, 1, x, t)
        return v.permute(0,2,3,1)

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




#GNO


class GNOLayer(nn.Module):
    """
    实现了GNO的核心层，基于方程 (10) [1]。
    v_{t+1}(x) = \sigma(W v_t(x) + Agg(Kernel(v_t(y))))
    """

    def __init__(self, in_channels, out_channels, N, activation=nn.GELU()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        # W: 作用于节点自身的线性变换 (对应公式中的 W v_t(x))
        self.W = nn.Linear(in_channels, out_channels)

        # Kernel Integration: 对应公式中的 sum(kappa * v_t(y))
        # 由于 forward 中没有显式传入坐标或邻接矩阵，为了无缝运行，
        # 这里使用一个全连接的空间混合矩阵 (Spatial Mixing) 来模拟全图积分。
        # 如果 N 很大，建议将其改为局部卷积或基于坐标的 Kernel。
        self.spatial_mixing = nn.Linear(N, N)
        self.kernel_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x shape: [B, Width, N, T]
        B, C, N, T = x.shape

        # 变换维度以便进行矩阵乘法: [B, T, N, C]
        x_in = x.permute(0, 3, 2, 1)

        # 1. 自身特征变换 (Local Term): W * v
        res = self.W(x_in)  # [B, T, N, Out_C]

        # 2. 邻域消息聚合 (Message Passing / Kernel Integration) [1]
        # 模拟对空间域 D 的积分/求和
        # 先对特征进行变换
        msg = self.kernel_lin(x_in)  # [B, T, N, Out_C]
        # 再在空间维度 N 上进行混合 (模拟图上的邻居聚合)
        msg = msg.permute(0, 1, 3, 2)  # [B, T, Out_C, N]
        msg = self.spatial_mixing(msg)  # [B, T, Out_C, N]
        msg = msg.permute(0, 1, 3, 2)  # [B, T, N, Out_C]

        # 合并
        out = res + msg

        if self.activation is not None:
            out = self.activation(out)

        # 恢复维度: [B, Out_C, N, T]
        return out.permute(0, 3, 2, 1)


class GNO2d(FNOBase):
    def __init__(
            self,
            N: int,
            T: int,
            input: int,
            width: int,
            energy_splits: List[float],  # 保留参数以兼容接口
            out_dim: int = 1,
            beta: float = -1e-2,
            delta: float = 1e-1,
            num_spectral_layers: int = 2,
            fft_norm: str = "backward",
            activation: str = "ReLU",  # 注意：这里通常传入字符串或类
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
        # 初始化父类，保持参数一致
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

        modes_t = T // 2
        self.width = width
        self.out_steps = output_steps
        self.debug = debug

        # 激活函数处理
        if isinstance(activation, str):
            if activation == "ReLU":
                act_func = nn.ReLU()
            elif activation == "GELU":
                act_func = nn.GELU()
            else:
                act_func = nn.ReLU()
        else:
            act_func = activation

        # 1. Lifting Operator (保持不变，用于特征映射)
        self.lifting_operator = LiftingOperator(
            input,
            width,
            N // 4,
            modes_t,
            input_shape=(N, T),
            latent_steps=latent_steps,
            norm=fft_norm,
            beta=beta,
            activation=activation,
            spatial_random_feats=spatial_random_feats,
            channel_expansion=channel_expansion,
            nonlinear=lift_activation,
        )

        assert num_spectral_layers > 1
        # 注意：原本这里是 spectral_layers，现在我们替换为 GNOLayers
        # 文献 [1] 指出 GNO 使用多层消息传递图网络
        self.gno_layers = nn.ModuleList()
        for _ in range(num_spectral_layers - 1):
            self.gno_layers.append(
                GNOLayer(width, width, N, activation=act_func)
            )

        # 投影层 (Projection / Decoding)
        self.reduction = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, v: torch.Tensor, eigvecs: torch.Tensor, eigval: torch.Tensor, time_feats: torch.Tensor,
                out_steps=None) -> torch.Tensor:
        """
        参数保持与 GTFNO2d 完全一致，以便直接替换。
        v: real [B, N, T, F_in]
        eigvecs: [N, K] (GNO算法中通常不直接使用谱特征，但保留接口)
        """
        if out_steps is None:
            out_steps = self.out_steps if self.out_steps is not None else v.size(-1)

        # 维度调整: [B, N, T, F] -> [B, F, N, T]
        v = rearrange(v, "b x t f -> b f x t")

        # Lifting: [B, F, N, T] -> [B, Width, N, T]
        v = self.lifting_operator(v, eigvecs)

        # GNO Iterative Layers (Message Passing)
        # 替代了原有的 spectral_conv + mlp + w 结构
        # 这里的每一层对应公式 (10) 的一次迭代更新 [1]
        for layer in self.gno_layers:
            v = layer(v)

        # Projection to Output
        v = self.reduction(v)  # (B, Width, N, T) -> (B, 1, N, T)

        return v.permute(0, 2, 3, 1)  # -> [B, N, T, 1]

##Geo-FNO
class LatentSpectralConv3d(nn.Module):
    """
    修正版：使用分离的实部和虚部权重，避免 Adam 优化器在处理 cfloat 参数时的维度报错。
    """

    def __init__(self, in_channels, out_channels, modes_h, modes_w, modes_t):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_w
        self.modes_t = modes_t

        scale = (1 / (in_channels * out_channels))

        # --- 核心修改：将复数权重拆分为实部和虚部定义 ---
        # 形状为 (in, out, mh, mw, mt, 2)，最后一位 0 是实部，1 是虚部
        # 这种方式对所有 PyTorch 版本和优化器都最安全

        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_h, modes_w, modes_t, 2, dtype=torch.float32))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_h, modes_w, modes_t, 2, dtype=torch.float32))
        self.weights3 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_h, modes_w, modes_t, 2, dtype=torch.float32))
        self.weights4 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_h, modes_w, modes_t, 2, dtype=torch.float32))

    def get_complex_weight(self, weight_tensor):
        # 将 (..., 2) 的 float 张量转换为 complex 张量
        return torch.view_as_complex(weight_tensor)

    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x, y, t), (in_channel, out_channel, x, y, t) -> (batch, out_channel, x, y, t)
        # 使用 einsum 进行复数乘法
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # 1. 3D FFT
        # x: [B, C, H, W, T] -> FFT -> [B, C, H, W, T//2 + 1] (Complex)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # 2. 准备输出容器
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # 3. 获取复数权重 (动态合成)
        w1 = self.get_complex_weight(self.weights1)
        w2 = self.get_complex_weight(self.weights2)
        w3 = self.get_complex_weight(self.weights3)
        w4 = self.get_complex_weight(self.weights4)

        # 4. 频率混合 (Corner Modes)
        # 注意：这里使用了切片操作，确保维度匹配
        # corner 1
        out_ft[:, :, :self.modes_h, :self.modes_w, :self.modes_t] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_h, :self.modes_w, :self.modes_t], w1)

        # corner 2
        out_ft[:, :, -self.modes_h:, :self.modes_w, :self.modes_t] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_h:, :self.modes_w, :self.modes_t], w2)

        # corner 3
        out_ft[:, :, :self.modes_h, -self.modes_w:, :self.modes_t] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_h, -self.modes_w:, :self.modes_t], w3)

        # corner 4
        out_ft[:, :, -self.modes_h:, -self.modes_w:, :self.modes_t] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_h:, -self.modes_w:, :self.modes_t], w4)

        # 5. Inverse 3D FFT
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class GeoFNO2d(FNOBase):
    def __init__(
            self,
            N: int,
            T: int,
            input: int,
            width: int,
            energy_splits: List[float],
            out_dim: int = 1,
            beta: float = -1e-2,
            delta: float = 1e-1,
            num_spectral_layers: int = 2,
            fft_norm: str = "backward",
            activation: str = "ReLU",
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

        modes_t = T // 2
        self.width = width
        self.out_steps = output_steps
        self.debug = debug
        self.N = N

        # Geo-FNO: Latent Grid Configuration
        # 设定潜在网格大小
        self.latent_h = int(N ** 0.5)
        self.latent_w = int(N ** 0.5)
        if self.latent_h * self.latent_w < N:
            self.latent_h += 1

        self.latent_dim = self.latent_h * self.latent_w

        # Geo-Encoder / Decoder (Linear Deformation)
        self.geo_encoder = nn.Linear(N, self.latent_dim)
        self.geo_decoder = nn.Linear(self.latent_dim, N)

        # FFT Modes
        modes_h = self.latent_h // 2
        modes_w = self.latent_w // 2

        # 激活函数
        if isinstance(activation, str):
            if activation == "ReLU":
                act_func = nn.ReLU()
            elif activation == "GELU":
                act_func = nn.GELU()
            else:
                act_func = nn.ReLU()
        else:
            act_func = activation

        self.lifting_operator = LiftingOperator(
            input,
            width,
            N // 4,
            modes_t,
            input_shape=(N, T),
            latent_steps=latent_steps,
            norm=fft_norm,
            beta=beta,
            activation=activation,
            spatial_random_feats=spatial_random_feats,
            channel_expansion=channel_expansion,
            nonlinear=lift_activation,
        )

        assert num_spectral_layers > 1
        num_spectral_layers -= 1

        # 使用修正后的 LatentSpectralConv3d
        self.spectral_conv = nn.ModuleList(
            [
                LatentSpectralConv3d(width, width, modes_h, modes_w, modes_t)
                for _ in range(num_spectral_layers)
            ]
        )

        self.mlp = nn.ModuleList(
            [nn.Sequential(
                nn.Conv3d(width, width * channel_expansion, 1),
                act_func,
                nn.Conv3d(width * channel_expansion, width, 1)
            ) for _ in range(num_spectral_layers)]
        )

        self.w = nn.ModuleList(
            [nn.Conv3d(width, width, 1) for _ in range(num_spectral_layers)]
        )

        self.activations = nn.ModuleList(
            [act_func for _ in range(num_spectral_layers)]
        )

        self.reduction = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, v: torch.Tensor, eigvecs: torch.Tensor, eigval: torch.Tensor, time_feats: torch.Tensor,
                out_steps=None) -> torch.Tensor:
        if out_steps is None:
            out_steps = self.out_steps if self.out_steps is not None else v.size(-1)

        # [B, N, T, F] -> [B, F, N, T] -> [B, Width, N, T]
        v = rearrange(v, "b x t f -> b f x t")
        v = self.lifting_operator(v, eigvecs)

        # 1. Geo-Encoding: N -> Latent Grid
        B, C, N, T = v.shape
        v = v.permute(0, 1, 3, 2)  # [B, C, T, N]
        v = self.geo_encoder(v)  # [B, C, T, Latent_Dim]

        # Reshape to 3D Grid: [B, C, H, W, T]
        # 这里的 contiguous() 很关键，防止 reshape 导致内存不连续引发错误
        v = v.view(B, C, T, self.latent_h, self.latent_w)
        v = v.permute(0, 1, 3, 4, 2).contiguous()  # [B, C, H, W, T]

        # 2. FNO Processing
        for conv, mlp, w, nonlinear in zip(self.spectral_conv, self.mlp, self.w, self.activations):
            x1 = conv(v)
            x1 = mlp(x1)
            x2 = w(v)
            v = x1 + x2
            v = nonlinear(v)

        # 3. Geo-Decoding: Latent Grid -> N
        # [B, C, H, W, T] -> [B, C, T, N]
        v = v.permute(0, 1, 4, 2, 3).contiguous()  # [B, C, T, H, W]
        v = v.view(B, C, T, -1)  # [B, C, T, Latent_Dim]
        v = self.geo_decoder(v)  # [B, C, T, N]

        # Projection
        v = v.permute(0, 1, 3, 2).contiguous()  # [B, C, N, T]
        v = self.reduction(v)  # [B, 1, N, T]

        return v.permute(0, 2, 3, 1)  # [B, N, T, 1]
