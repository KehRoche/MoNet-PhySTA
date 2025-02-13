from nni.experiment import Experiment
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# more window_size, prior:gru,linear could remove'Inception_1d''Linear'
search_space = {
    # 'prior_method': {'_type': 'choice', '_value': ['GRU']},
    # 'series_method': {'_type': 'choice', '_value': ['MultiheadDilatedAttention']},
    'anormly_ratio': {"_type": "quniform", "_value": [0.6, 0.8, 0.001]},
    'gru_dropout': {"_type": "quniform", "_value": [0.6, 0.7, 0.001]},
    'window_size': {"_type": 'choice', "_value": [256,512]},
    'd_model': {"_type": 'choice', "_value": [128,256]},
    #,
    'dataset': {'_type': 'choice', '_value': ['SMAP','SMD','PSM','NIPS_TS_Water''MSL','NIPS_TS_Swan']}

}

experiment = Experiment('local')
experiment.config.trial_command = 'python main.py  --num_epochs 3 --batch_size 256  --mode train'
experiment.config.trial_code_directory = '.'

experiment.config.search_space = search_space
# GridSearch
experiment.config.tuner.name = 'GridSearch'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 1000
experiment.config.trial_concurrency = 1
experiment.config.trial_gpu_number = 1
experiment.config.training_service.use_active_gpu = True

experiment.run(8080)
# input('Press enter to quit')
experiment.stop()
experiment.view()

