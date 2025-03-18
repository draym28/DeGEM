import os

SEED = 28
DATA_DIR = os.path.abspath(os.path.dirname(__file__)) + '/_data/'


# for twitch
test_sg = ['ES', 'FR', 'RU']

# for arxiv
test_year = [2018, 2019, 2020]


title = {
    'detect': ['auroc', 'aupr', 'fpr', 'test_acc'], 
}


def default_config():
    config = {
        # setup and protocol
        'mode': 'detect', 
        'ood_type': 'structure', 

        'num_runs': 1, 
        'num_epochs': 200, 

        # dataset
        'train_prop': .1, 
        'valid_prop': .1, 

        'method': 'degem', 

        # pre propagation
        'preprop_K': 5, 
        'beta': 0.5,

        # cl network
        'cl_model': 'dgi', 
        'cl_hdim': 512, 

        # energy network
        'backbone': 'mlp',  # mlp, bilinear
        'hidden_channels': 64, 
        'num_layers': 2, 

        # classifier
        'lam': 0.1, # classification loss weight

        'rho': 1., 
        'gamma': 1., 

        # ebm hyper
        'coef_reg': 1., 
        'mcmc_steps': 20,  # number of MCMC sampling, default 20
        'mcmc_step_size': 1.,  # grad coef in MCMC sampling
        'mcmc_noise': 0.005,  # noise in MCMC sampling
        'max_buffer_vol': 2,  # max buffer len is $max_buffer_vol$ times of num nodes

        # training
        'lr': 0.001, 
        'wd': 5e-4, 
        'dropout': 0.5, 

        # for eval, fixed
        'eval_epochs': 100, 
        'lr_evals': [0.05, 0.01, 0.001, 0.0001], 
        'wd_evals': [0.0, 1e-3, 5e-4], 
        'lr_eval': 0.01, 
        'wd_eval': 0.
    }

    return config


def config(ds='cora'):
    config = default_config()

    return config