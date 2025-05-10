import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class DSCLayer(nn.Module):
    def __init__(self, input_dim, out_dim, kernel_size=4):
        super().__init__()
        self.Resnet = nn.Sequential(
            Conv1d(input_dim, input_dim, kernel_size=kernel_size, groups=input_dim, padding='same'),
            nn.GELU()
        )
        self.Conv_1x1 = nn.Sequential(
            Conv1d(input_dim, input_dim, kernel_size=1),
            nn.GELU()
        )
        self.bn = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        x = self.Conv_1x1(x + self.Resnet(x))
        b, n = x.shape[:2]
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = x.reshape([b, n, -1, x.shape[-1]])
        return x


class ImprovedDSCLayer(nn.Module):
    def __init__(self, input_dim, kernel_size=4):
        super().__init__()
        # 添加跨通道交互
        self.channel_mixer = Conv1d(input_dim, input_dim, kernel_size=1)
        self.depthwise = Conv1d(input_dim, input_dim, kernel_size=kernel_size,
                                   groups=input_dim, padding='same')
        self.activation = nn.GELU()
        # 使用自适应归一化
        self.norm = nn.InstanceNorm1d(input_dim)

    def forward(self, x):
        identity = x
        b, n,t,c = x.shape
        x = self.channel_mixer(x)  # 跨通道交互
        x = self.depthwise(x)
        x = self.activation(x)+ identity
        x = self.norm(x.reshape(b*n,t,c))
        return x.reshape(b,n,t,c)


class DynmiacGate(nn.Module):
    def __init__(self, input_dim,emd_dim):
        super().__init__()
        self.flow_gate = nn.Sequential(
            nn.Linear(emd_dim * 3, emd_dim),
            nn.GELU(),
            nn.Linear(emd_dim, 1)  # 参考内容中的sigmoid门控
        )

    def forward(self, local_fea,time_emd):
        #通过动态权重生成实现多模式解耦,引导模型在多模式中关注当前主导模式
        gate = torch.sigmoid(self.flow_gate(torch.cat([local_fea, time_emd], dim=-1)))
        local_fea = local_fea * gate
        return local_fea

class DTWSimilarity:
    def __init__(self, gamma=0.1, radius=5, batch_mode='auto'):
        """
        :param gamma: SoftDTW平滑系数（参考值0.1-1.0）
        :param radius: Sakoe-Chiba窗口半径
        :param batch_mode: 批处理模式选择[auto, cuda, memsave]
        """
        self.gamma = gamma
        self.radius = radius
        self.batch_mode = batch_mode

    def _compute_batch(self, X):
        """核心计算函数（已通过CUDA加速）"""
        N, T, F = X.shape
        sdtw = SoftDTW(gamma=self.gamma,
                       use_cuda=False)

        # 矩阵展开策略提升并行度
        X_exp = X.unsqueeze(1).expand(-1, N, T, F)
        X_ref = X.unsqueeze(0).expand(N, -1, T, F)

        # 分块计算防止显存溢出（当N>100时自动启用）
        if self.batch_mode == 'auto':
            block_size = 50 if N > 100 else N
            dist_matrix = torch.zeros(N, N, device=X.device)

            for i in range(0, N, block_size):
                for j in range(0, N, block_size):
                    end_i = min(i + block_size, N)
                    end_j = min(j + block_size, N)
                    current_block_size_i = end_i - i
                    current_block_size_j = end_j - j

                    # 提取当前块的数据
                    dist_block = sdtw(
                        X_exp[i:end_i, j:end_j].reshape(-1, T, F),
                        X_ref[i:end_i, j:end_j].reshape(-1, T, F)
                    )

                    # 将计算结果放入距离矩阵的相应位置
                    dist_matrix[i:end_i, j:end_j] = dist_block.view(current_block_size_i, current_block_size_j)
            return dist_matrix
        else:
            return sdtw(X_exp.reshape(N * N, T, F),
                        X_ref.reshape(N * N, T, F)).view(N, N)

    def __call__(self, X):
        """
        :param X: 输入张量 (B, N, T, F)
        :return: 相似度矩阵 (B, N, N)
        """
        B, N, T, F = X.shape

        # 多流并行计算
        streams = [torch.cuda.Stream() for _ in range(B)]
        results = torch.zeros(B, N, N, device=X.device)

        # 异步流水线处理
        with torch.no_grad():
            for b in range(B):
                with torch.cuda.stream(streams[b]):
                    # 标准化处理（提升数值稳定性）
                    X_norm = (X[b] - X[b].mean(1, keepdim=True)) / (X[b].std(1, keepdim=True) + 1e-6)
                    dist = self._compute_batch(X_norm)
                    results[b] = torch.sigmoid(dist * 0.1)  # 缩放系数控制梯度范围

        torch.cuda.synchronize()
        return results


class AdaptiveRevIN(nn.Module):
    def __init__(self, op_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=op_dim, hidden_size=op_dim*2)
        self.mu_layer = nn.Linear(op_dim*2, op_dim)
        self.sigma_layer = nn.Linear(op_dim*2, op_dim)

    def norm(self, x):
        # x: [B, N, T, C]
        context = self.lstm(x.mean(dim=2))[1][0]  # 时间维度聚合
        mu = self.mu_layer(context)
        sigma = self.sigma_layer(context)
        return (x - mu.unsqueeze(2)) / (sigma.unsqueeze(2) + 1e-6)

    def denorm(self, x_norm, original_x):
        # 反向计算使用原始数据的统计量
        return x_norm * original_x.std(dim=2, keepdim=True) + original_x.mean(dim=2, keepdim=True)

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,groups=1, kernel_size=1,padding='same', dropout=0.1, actv=True):
        super(Conv1d, self).__init__()
        self.convolution = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.actv = actv

    def forward(self, x):
        if len(x.shape) == 4:
            b, n = x.shape[:2]
            x = x.flatten(0, 1)
            y = self.convolution(x)
            y = y.reshape([b, n, -1, y.shape[-1]])
        else:
            y = self.convolution(x)
        if self.actv:
            y =  F.gelu(self.dropout(y))
        return y


def compute_laplacian(adj_matrix):
    """
    自适应计算拉普拉斯矩阵：无向图使用普通拉普拉斯，有向图使用磁性拉普拉斯
    :param adj_matrix: 邻接矩阵 (Tensor)
    :return: 特征向量和特征值
    """
    # 判断是否为无向图（邻接矩阵对称）
    is_undirected = torch.allclose(adj_matrix, adj_matrix.T, atol=1e-6)

    if is_undirected:
        return _compute_undirected_laplacian(adj_matrix)
    else:
        return _compute_directed_laplacian(adj_matrix)

def _compute_undirected_laplacian(adj_matrix):
    """
    计算无向图的普通拉普拉斯矩阵 L = D - A
    """
    # 计算度矩阵 D
    degrees = torch.sum(adj_matrix, dim=1)
    D = torch.diag_embed(degrees)

    # 拉普拉斯矩阵
    L = D - adj_matrix

    # 特征分解
    eigvals, eigvecs = torch.linalg.eigh(L)
    eigvecs = eigvecs / eigvecs.norm(dim=0, keepdim=True)  # 归一化
    return eigvecs, eigvals

def _compute_directed_laplacian(adj_matrix):
    """
    计算有向图的磁性拉普拉斯矩阵（保留原逻辑）
    """
    diff_matrix = adj_matrix - adj_matrix.T

    # 定义显著性阈值（基于差异的绝对值）
    threshold = torch.quantile(diff_matrix[diff_matrix != 0].abs(), 0.5)

    # 生成二进制方向矩阵
    dir_diff_matrix = torch.where(diff_matrix > threshold, diff_matrix, torch.zeros_like(diff_matrix))

    # 根据原始权重增强方向差异矩阵
    weight_threshold = torch.median(adj_matrix[adj_matrix != 0])
    weight_mask = (adj_matrix > weight_threshold).float()
    A_dir = dir_diff_matrix * weight_mask

    # 构造带相位的复数邻接矩阵
    q = 0.25  # 相位参数
    theta = torch.tensor(q * (2 * math.pi))
    phase = torch.complex(torch.cos(theta), torch.sin(theta))

    A_complex = adj_matrix * (1 + 0j)
    A_complex[A_dir > 0] *= phase

    # 计算磁性度矩阵
    D_complex = torch.diag_embed(torch.sum(A_complex, dim=1))

    # 构造磁性拉普拉斯矩阵
    L_mag = D_complex - 0.5 * (A_complex + A_complex.conj().transpose(-2, -1))

    # 特征分解
    eigvals, eigvecs = torch.linalg.eigh(L_mag)
    eigvecs = eigvecs / eigvecs.norm(dim=0, keepdim=True)  # 归一化
    return eigvecs, eigvals

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, hidden_dim, output_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=hidden_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=output_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        return hidden