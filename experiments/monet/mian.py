import os
import argparse
import numpy as np
import yaml
import sys
import torch.nn as nn

sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch

torch.set_num_threads(3)

from src.models.monet import MoNet
from src.engines.monet_engine  import MONET_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info, load_adj_from_pickle
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config(config_path):
    parser = get_public_config()
    parser.add_argument('--cl_epoch', type=int, default=3)
    parser.add_argument('--warm_epoch', type=int, default=30)
    parser.add_argument('--tpd', type=int, default=96)

    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = config['config1']
    # 将命令行参数与配置文件参数合并
    # 这里我们通过遍历命令行参数来覆盖配置文件的值
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    log_dir = './experiments/{}/{}/'.format(args.model_name, config['dataset'])
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(config)

    return config, log_dir, logger

def nas(config):
    from nni.nas.evaluator import FunctionalEvaluator
    import nni
    import nni.nas.strategy as strategy

    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
#'prior_method': 'MultiheadDilatedAttention', 'series_method': 'Inception_1d',
    params = {
        'anormly_ratio': 0.6,
        'gru_dropout':0.5,
        'd_model':128,
        'window_size': 128,
        'dataset': 'MSL'
    }

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)

    # config.prior_method = params['prior_method']
    # config.series_method = params['series_method']
    config.win_size = params['window_size']
    config.gru_dropout = params['gru_dropout']
    config.d_model = params['d_model']
    config.anormly_ratio = params['anormly_ratio']
    config.dataset = params['dataset']
    config.data_path = params['dataset']
    if config.dataset == 'MSL':
        config.input_c = 55
    elif config.dataset == 'SMD':
        config.input_c = 38
    elif config.dataset == 'PSM':
        config.input_c = 25
    elif config.dataset == 'SMAP':
        config.input_c = 25
    elif config.dataset == 'NIPS_TS_Swan':
        config.input_c = 38
    elif config.dataset == 'NIPS_TS_Water':
        config.input_c = 9



    solver = Solver(vars(config))
    solver.train()
    _,precision, recall,f1_score=solver.test()
    nni.report_final_result(f1_score)
    precision = "{:.3f}".format(precision*100)
    recall = "{:.3f}".format(recall*100)

def init_weights(m):
    """分层初始化：卷积、BN、线性层差异化处理"""
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:  # 偏置项非空时初始化为0
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)  # BN层gamma初始化为1
        if hasattr(m, 'bias') and m.bias is not None:  # 关键修改
            nn.init.zeros_(m.bias)  # BN层beta初始化为0
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if hasattr(m, 'bias') and m.bias is not None:  # 关键修改
            nn.init.normal_(m.bias, mean=0, std=0.01)
    elif isinstance(m, nn.Parameter):
        # 针对特定参数类型初始化
        if 'weight' in m.name:
            nn.init.kaiming_normal_(m, mode='fan_in', nonlinearity='relu')
        elif 'coff' in m.name:
            nn.init.constant_(param, 1.0)
def main():
    config, log_dir, logger = get_config('config.yaml')

    # 使用字典的方式来访问和修改
    set_seed(config['seed'])  # 使用 config 字典中的 'seed' 键

    device = torch.device(config['device'])  # 使用 config 字典中的 'device' 键

    # 获取数据集相关信息
    data_path, adj_path, node_num = get_dataset_info(config['dataset'])
    if(config['dataset'] == 'PEMS-BAY'):
        A = load_adj_from_pickle(adj_path)[2]
    else:
        A = load_adj_from_numpy(adj_path)
    config['adj'] = torch.tensor(A).to(device)
    location_path = data_path + '/location.npy'
    location = np.load(location_path)
    config['location'] = torch.tensor(location).to(device)
    dataloader, scaler = load_dataset(data_path, config, logger)

    cl_step = config['cl_epoch'] * dataloader['train_loader'].num_batch
    warm_step = config['warm_epoch'] * dataloader['train_loader'].num_batch

    model = MoNet(input_dim=config['input_dim'],
                  output_dim=config['output_dim'],
                  model_config=config  # 将 config 传递给模型
                  )

    model.apply(init_weights)
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lrate'], weight_decay=config['wdecay'], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 38, 46, 54, 62, 70, 80], gamma=0.5)

    engine = MONET_Engine(device=device,
                          model=model,
                          dataloader=dataloader,
                          scaler=scaler,
                          sampler=None,
                          loss_fn=loss_fn,
                          lrate=config['lrate'],
                          optimizer=optimizer,
                          scheduler=scheduler,
                          clip_grad_value=config['clip_grad_value'],
                          max_epochs=config['max_epochs'],
                          patience=config['patience'],
                          log_dir=log_dir,
                          logger=logger,
                          seed=config['seed'],
                          cl_step=cl_step,
                          warm_step=warm_step,
                          horizon=config['horizon'],
                          tempvar_penalty=config['temp_penalty'],
                          spatialvar_penalty=config['spital_penalty']
                          )

    # 根据运行模式选择训练或评估
    if config['mode'] == 'train':
        engine.train()
    else:
        engine.evaluate(config['mode'])
if __name__ == "__main__":
    main()