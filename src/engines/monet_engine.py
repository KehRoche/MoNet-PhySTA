import torch
import numpy as np
import time
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse


class MONET_Engine(BaseEngine):
    def __init__(self, cl_step, warm_step, horizon, **args):
        super().__init__(**args)
        self._cl_step = cl_step
        self._warm_step = warm_step
        self._horizon = horizon
        self._cl_len = 0


    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        total_batches = self._dataloader['train_loader'].num_batch
        log_every = max(1, min(100, total_batches // 10))
        epoch_start = time.time()
        for batch_idx, (X, label) in enumerate(self._dataloader['train_loader'].get_iterator(), start=1):
            batch_start = time.time()
            self._optimizer.zero_grad()
            X, label = self._to_device(self._to_tensor([X, label]))
            # # 获取节点数量 N（假设 X 形状为 [B, N, ...]）


            # 对选中的节点，在特征和标签上全部置零
            # 如果 X 维度为 [B, N, F] 或 [B, N, T, F]，请相应调整下标
            X[:,:,self.mask_idx, ...] = 0
            label[:,:, self.mask_idx, ...] = 0

            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                self._logger.info('Mask value: %s', mask_value)

            self._iter_cnt += 1
            if self._iter_cnt < self._warm_step:
                self._cl_len = self._horizon
            elif self._iter_cnt == self._warm_step:
                self._cl_len = 1
            else:
                if (self._iter_cnt - self._warm_step) % self._cl_step == 0 and self._cl_len < self._horizon:
                    self._cl_len += 1

            pred = pred[:, :self._cl_len, :, :]
            label = label[:, :self._cl_len, :, :]


            # temporal_var = torch.var(pred, dim=1, keepdim=False)  # 移除时间维度
            # temporal_penalty = torch.mean(temporal_var) * self.tempvar_penalty
            # spatial_var = torch.var(pred, dim=2, keepdim=False)  # 移除空间维度
            # spatial_penalty = torch.mean(spatial_var) * self.spatialvar_penalty
            #+temporal_penalty+spatial_penalty
            loss = self._loss_fn(pred, label, mask_value,self.mask_idx)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()
            
            

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches:
                elapsed = time.time() - epoch_start
                self._logger.info(
                    'Train batch %d/%d, Loss: %.4f, RMSE: %.4f, MAPE: %.4f, Batch Time: %.2fs, Elapsed: %.2fs',
                    batch_idx, total_batches, train_loss[-1], train_rmse[-1], train_mape[-1],
                    time.time() - batch_start, elapsed
                )
        #writer.close()
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)
