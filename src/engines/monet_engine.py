import torch
import numpy as np
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse
import torch.profiler
from torch.utils.tensorboard import SummaryWriter


class MONET_Engine(BaseEngine):
    def __init__(self, cl_step, warm_step, horizon, **args):
        super(MONET_Engine, self).__init__(**args)
        self._cl_step = cl_step
        self._warm_step = warm_step
        self._horizon = horizon
        self._cl_len = 0


    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        writer = SummaryWriter(log_dir='runs/temp_module')

        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('check mask value', mask_value)

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

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            # for name, param in self.model.named_parameters():
            #     writer.add_histogram(f'value/{name}', param, self._iter_cnt)
            #     if param.grad is not None and param.grad.numel() > 0 and param.grad.abs().sum() > 0:
            #         # 将梯度记录到TensorBoard，使用scalars来记录每一层的梯度信息
            #         writer.add_histogram(f'grad/{name}', param.grad, self._iter_cnt)
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
        writer.close()
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)