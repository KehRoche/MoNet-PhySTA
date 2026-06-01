import os
import pickle
import torch
import numpy as np
from pathlib import Path

class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = np.asarray(self.idx[start_ind: end_ind]).reshape(-1)

                x = self.data[idx_ind[:, None] + self.x_offsets[None, :], :, :]
                y = self.data[idx_ind[:, None] + self.y_offsets[None, :], :, :1]

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        return data * std + mean


def load_dataset_largest(data_path, args, logger):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
    logger.info('Data shape: ' + str(ptr['data'].shape))
    
    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))
        dataloader[cat + '_loader'] = DataLoader(ptr['data'][..., :args.input_dim], idx, \
                                                 args.seq_len, args.horizon, args.bs, logger)

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler


def load_dataset(data_path, config, logger):
    ptr = np.load(os.path.join(data_path, 'his.npz'))
    logger.info('Data shape: ' + str(ptr['data'].shape))
    input_dim = config['input_dim']
    seq_len = config['seq_len']
    horizon = config['horizon']
    bs = config['bs']

    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path,  'idx_' + cat + '.npy'))
        dataloader[cat + '_loader'] = DataLoader(ptr['data'][..., :input_dim], idx, \
                                                 seq_len, horizon, bs, logger)

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = Path(__file__).resolve().parents[2] / 'data'
    d = {
         'CA': [base_dir / 'ca', base_dir / 'ca' / 'ca_rn_adj.npy', 8600],
         'GLA': [base_dir / 'gla', base_dir / 'gla' / 'gla_rn_adj.npy', 3834],
         'GBA': [base_dir / 'gba', base_dir / 'gba' / 'gba_rn_adj.npy', 2352],
         'SD': [base_dir / 'sd', base_dir / 'sd' / 'sd_rn_adj.npy', 716],
         'PEMS-BAY': [base_dir / 'PEMS-BAY', base_dir / 'PEMS-BAY' / 'adj_mx_bay.pkl', 325],
         'PEMS08': [base_dir / 'PEMS08', base_dir / 'PEMS08' / 'adj_mx_08_distance.npy', 170],
         'KnowAir': [base_dir / 'KnowAir', base_dir / 'KnowAir' / 'adj_matrix.npy', 184],
         'BJAir': [base_dir / 'BJAir', base_dir / 'BJAir' / 'BJAir.npy', 35]
    }
    assert dataset in d.keys()
    data_path, adj_path, node_num = d[dataset]
    return str(data_path), str(adj_path), node_num
