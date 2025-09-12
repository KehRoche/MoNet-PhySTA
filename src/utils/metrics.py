import torch

def masked_mse(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=torch.nan, mask_idx=None):
    """
    计算 masked MAE，仅对非 null_val 且不在 mask_idx 中的节点计算。
    支持 preds, labels 为 (B, T, N, C) 的张量。

    参数:
    - preds: Tensor，形状为 (B, T, N, C)
    - labels: Tensor，形状为 (B, T, N, C)
    - null_val: float，NaN 或填充值，默认 NaN
    - mask_idx: list[int] or Tensor[int]，需要被 mask 的节点索引 N 维度

    返回:
    - loss: scalar，masked MAE
    """
    # 1. 基础有效性掩码：对 null_val 屏蔽
    if isinstance(null_val, float) and torch.isnan(torch.tensor(null_val)):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    # 2. 节点维度的屏蔽（第 N 维）
    if mask_idx is not None:
        # 构造 shape 为 (1, 1, N, 1) 的广播掩码
        node_mask = torch.ones((1, 1, labels.shape[2], 1), dtype=torch.bool, device=labels.device)
        node_mask[:, :, mask_idx, :] = False
        mask &= node_mask  # 广播乘法屏蔽整个 node 的所有 channel

    # 3. 安全防护：有效元素为 0 时避免除 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=labels.device)

    # 4. MAE 计算
    loss = torch.abs(preds - labels)
    loss = loss * mask  # 元素级屏蔽
    return loss.sum() / mask.sum()


def masked_mape(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    return mae, mape, rmse