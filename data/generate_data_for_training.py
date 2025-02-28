import os
import argparse
import numpy as np
import pandas as pd

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min = None
        self.max = None
        self.feature_range = feature_range

    def fit(self, data):
        """Compute the minimum and maximum to be used for later scaling."""
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        """Scale the data to the specified range."""
        data_scaled = (data - self.min) / (self.max - self.min)
        return data_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, data):
        """Fit to data, then transform it."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """Inverse the scaling."""
        data_rescaled = (data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return data_rescaled * (self.max - self.min) + self.min

def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    num_feat=1
    num_samples, num_nodes, _ = data.shape
    # add_time_in_day = False
    # add_day_in_week = False
    feature_list = [data[..., 0:num_feat]]
    if add_time_in_day:
        # numerical time_in_day
        time_ind = [i%288 / 288 for i in range(num_samples)]
        time_ind = np.array(time_ind)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if add_day_in_week:
        # numerical day_in_week
        day_in_week = [(i // 288)%7 for i in range(num_samples)]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)

    data = np.concatenate(feature_list, axis=-1)
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print('idx min & max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)
    return data, idx
def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    
    feature_list = [data]
    if add_time_of_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day)
    if add_day_of_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled / 7
        feature_list.append(day_of_week)

    data = np.concatenate(feature_list, axis=-1)
    
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print('idx min & max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)
    return data, idx


def generate_train_val_test(args):
    years = args.years.split('_')

    df = pd.DataFrame()
    # for y in years:
    #     df_tmp = pd.read_hdf(args.dataset + '/' + args.dataset + '_his_' + y + '.h5')
    #     df = df.append(df_tmp)

    df = pd.read_hdf('data/'+args.dataset + '/' + args.dataset  + '.h5')
    #traffic_df_filename = 'data/PEMS04/PEMS04.npz'
    #df = np.load(traffic_df_filename)['data']

    print('original data shape:', df.shape)

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    data, idx = generate_data_and_idx(df, x_offsets, y_offsets, args.tod, args.dow)
    #data, idx = generate_graph_seq2seq_io_data(df, x_offsets, y_offsets, args.tod, args.dow)
    print('final data shape:', data.shape, 'idx shape:', idx.shape)

    num_samples = len(idx)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)

    # split idx
    idx_train = idx[:num_train]
    idx_val = idx[num_train: num_train + num_val]
    idx_test = idx[num_train + num_val:]

    # normalize
    x_train = data[:idx_val[0] - args.seq_length_x, :, 0] 
    #scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    data[..., 0] = scaler.transform(data[..., 0])

    # save
    out_dir = 'data/'+args.dataset + '/' + args.years
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(os.path.join(out_dir, 'minmax_his.npz'), data=data, mean=scaler.min, std=scaler.max)

    # np.save(os.path.join(out_dir, 'idx_train'), idx_train)
    # np.save(os.path.join(out_dir, 'idx_val'), idx_val)
    # np.save(os.path.join(out_dir, 'idx_test'), idx_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PEMS-BAY', help='dataset name')
    parser.add_argument('--years', type=str, default='', help='if use data from multiple years, please use underline to separate them, e.g., 2018_2019')
    parser.add_argument('--seq_length_x', type=int, default=12, help='sequence Length')
    parser.add_argument('--seq_length_y', type=int, default=12, help='sequence Length')
    parser.add_argument('--tod', type=int, default=1, help='time of day')
    parser.add_argument('--dow', type=int, default=1, help='day of week')
    
    args = parser.parse_args()
    generate_train_val_test(args)
