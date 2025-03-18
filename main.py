import os
from termcolor import cprint
from tqdm import tqdm
import json
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F

from data_utils import evaluate_detect, rand_splits
from dataset import load_dataset, prepare_dataset

from model import DeGEM
import utils as u
import config as c


device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
conv_cuda = 1


def eval(model:DeGEM, state_dict, dataset, config):

    edge_index, y = dataset.edge_index, dataset.y
    train_idx = dataset.splits['train']
    test_idx = dataset.splits['test']
    model.eval()
    embed = model.encoder.embed(dataset).detach()

    best_acc = 0.
    for lr in config['lr_evals']:
        for wd in config['wd_evals']:
            model.load_state_dict(state_dict)
            opt = torch.optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=wd)

            for _ in tqdm(range(config['eval_epochs']), ncols=70):
                model.classifier.train()
                opt.zero_grad()

                logits = model.classifier(embed[train_idx])
                loss = F.cross_entropy(logits, y[train_idx])

                loss.backward()
                opt.step()

            logits = model.classifier(embed[test_idx])
            acc = (logits.argmax(dim=1) == y[test_idx]).float().mean().item() * 100

            if acc > best_acc:
                best_acc = acc
                config['lr_eval'] = lr
                config['wd_eval'] = wd

    return best_acc, config


def main(dataname='cora', config=None, log_root='log'):

    torch.cuda.reset_peak_memory_stats()

    u.set_seed(c.SEED)
    os.makedirs(log_root, exist_ok=True)
    folder_name = os.getcwd().split(os.sep)[-1]

    if config is None:
        config = c.config(dataname)

    cprint(f'{folder_name} {u.save_file_name(dataname, config)}', 'red')

    ### Load and preprocess data ###
    # id data, ood data for training, ood data for test
    # dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(dataname, c.DATA_DIR, config)
    dataset_ind, _, dataset_ood_te = load_dataset(dataname, c.DATA_DIR, config)

    dataset_ind.y = dataset_ind.y.squeeze()
    # dataset_ood_tr.y = dataset_ood_tr.y.squeeze()
    if isinstance(dataset_ood_te, list):
        for data in dataset_ood_te:
            data.y = data.y.squeeze()
    else:
        dataset_ood_te.y = dataset_ood_te.y.squeeze()

    ### get splits for all runs ###
    if dataname in ['cora', 'citeseer', 'pubmed']:
        pass
    else:
        dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=config['train_prop'], valid_prop=config['valid_prop'])

    ### print dataset info ###
    in_channels = dataset_ind.x.shape[1]  # original dimension
    num_classes = dataset_ind.y.unique().shape[0]  # num classes

    print(f"ind dataset {dataname}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
        + f"feats {in_channels}")
    # print(f"ood tr dataset {dataname}: all nodes {dataset_ood_tr.num_nodes} | centered nodes {dataset_ood_tr.node_idx.shape[0]} | edges {dataset_ood_tr.edge_index.size(1)}")
    if isinstance(dataset_ood_te, list):
        for i, data in enumerate(dataset_ood_te):
            print(f"ood te dataset {i} {dataname}: all nodes {data.num_nodes} | centered nodes {data.node_idx.shape[0]} | edges {data.edge_index.size(1)}")
    else:
        print(f"ood te dataset {dataname}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")


    ### load method ###
    model = DeGEM(in_channels, num_classes, config)

    dataset_ind = dataset_ind.to(device)
    # dataset_ood_tr = dataset_ood_tr.to(device)
    if config["mode"] == 'detect':
        if isinstance(dataset_ood_te, list):
            dataset_ood_te = [d.to(device) for d in dataset_ood_te]
        else:
            dataset_ood_te = dataset_ood_te.to(device)
    prepare_dataset(dataset_ind, device, model, cuda=conv_cuda)
    prepare_dataset(dataset_ood_te, device, model, cuda=conv_cuda)


    model.train()

    def _get_metric_for_best_epoch(result):
        # result: [(auroc(↑), aupr(↑), fpr(↓)) x #sg, acc(↑), loss(↓)]
        if dataname in ['twitch', 'arxiv']:
            tmp = 0
            for i in range(len(result) // 3):
                tmp = tmp + result[i*3]
            return tmp + result[-2]
        else:
            return result[0] + result[-2]

    ### Training loop ###
    results_total = []
    save_file = {'results': {}, 'config': {}}
    model.config = config
    for run in range(config['num_runs']):
        model.reset_parameters()
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['lr'], weight_decay=config['wd'])

        # results = []
        best_result = None
        best_metric = 0.
        best_model_sd = None
        for _ in tqdm(range(config['num_epochs']), ncols=70, unit='epoch'):
            model.train()
            optimizer.zero_grad()

            loss = model.loss_compute(dataset_ind, None)
            loss.backward()
            optimizer.step()

            # result: [(auroc(↑), aupr(↑), fpr(↓)) x #sg, acc(↑), loss(↓)]
            result = evaluate_detect(model, dataset_ind, dataset_ood_te)

            metric = _get_metric_for_best_epoch(result)  # auroc + acc
            if metric > best_metric:
                best_result = result.copy()
                best_metric = metric
                best_model_sd = deepcopy(model.state_dict())

        final_results = (100 * torch.tensor(best_result)).tolist()

        # eval ID acc
        model.load_state_dict(best_model_sd)
        test_acc, config = eval(model, best_model_sd, dataset_ind, config)
        if test_acc > final_results[-2]:
            final_results[-2] = test_acc

        results_total.append(final_results)
        print_str = u.print_str(dataname, run, final_results, config)
        cprint(print_str, 'green')

    results_data = u.save_file_data(dataname, results_total, config)
    save_file = {'results': results_data, 'config': config}
    save_file_name = u.save_file_name(dataname, config)
    save_file_path = f'{log_root}/results_{save_file_name}.json'
    with open(save_file_path, 'w') as f:
        json.dump(save_file, f, indent=4)

    print_str = u.print_str(dataname, run, np.mean(results_total, axis=0).tolist(), config, final=True)
    cprint(print_str, 'green')

    return u.return_things(save_file), results_data


if __name__ == "__main__":

    dataname_list = ['cora', 'amazon-photo', 'twitch', 'arxiv', 'chameleon', 'actor', 'cornell']
    ood_type_list = ['structure', 'feature', 'label']

    for dataname in dataname_list:
        config = c.config(dataname)
        if dataname in ['twitch', 'arxiv']:
            results, save_file = main(dataname=dataname, config=config, log_root=f'log_K{config["preprop_K"]}_ldsteps{config["mcmc_steps"]}')

        else:
            for ood_type in ood_type_list:
                config['ood_type'] = ood_type
                results, save_file = main(dataname=dataname, config=config, log_root=f'log_K{config["preprop_K"]}_ldsteps{config["mcmc_steps"]}')