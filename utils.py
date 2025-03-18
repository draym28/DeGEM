import os
import random
import numpy as np
import torch
import config as c


def set_seed(seed=c.SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def study_name(ds, config):
    name = f"{config['mode']}_{ds}"
    if ds not in ['twitch', 'arxiv']:
        name += f"_{config['ood_type']}"
    
    return name


def storage_name(ds, config):
    name = f"{config['method']}"
    name += f"_{config['cl_model']}_{config['backbone']}_{config['mode']}"

    return name


def save_file_name(ds, config):
    name = f"{config['method']}"
    name += f"_{config['cl_model']}_{config['backbone']}_{config['mode']}_{ds}"
    if ds not in ['twitch', 'arxiv']:
        name += f"_{config['ood_type']}"

    return name


def print_str(ds, run, results, config, final=False):
    title = c.title[config['mode']]
    num_values = len(title) - 1
    if final:
        s = f"Final\n"
    else:
        s = f"Run {run}\n"

    if config['mode'] == 'classify':
        for i, t in enumerate(title):
            s += f'{t}: {results[i]:.2f} '

    else:
        if ds not in ['twitch', 'arxiv']:
            for i, t in enumerate(title):
                s += f'\t| {t}: {results[i]:.2f}'
        else:
            ood_te_set = c.test_sg if ds == 'twitch' else c.test_year
            for n, ood_te in enumerate(ood_te_set):
                s += f"\t{ood_te} "
                for i, t in enumerate(title[:-1]):
                    s += f'|{t}: {results[n*num_values+i]:.2f} '
                s += '\n'
            s += f'\t{title[-1]}: {results[-2]:.2f} val_loss: {results[-1]:.2f}'

    if final:
        s += '\n'

    return s


def save_file_data(ds, results_total, config):
    data = {}
    title = c.title[config['mode']]
    results_total = torch.tensor(results_total)

    if config['mode'] == 'classify' or (config['mode'] == 'detect' and ds not in ['twitch', 'arxiv']):
        for r in range(results_total.shape[0]):
            data[f"run {r}"] = {}
            for i, t in enumerate(title):
                data[f"run {r}"][t] = results_total[r, i].item()
        results_mean = results_total.mean(dim=0)
        data["final"] = {}
        for i, t in enumerate(title):
            data["final"][t] = results_mean[i].item()

    elif config['mode'] == 'detect' and ds in ['twitch', 'arxiv']:

        ood_te_set = c.test_sg if ds == 'twitch' else c.test_year
        num_values = len(title) - 1
        num_set = len(ood_te_set)
        results_total_overall = torch.zeros([results_total.shape[0], num_values])
        for i in range(num_set):
            results_total_overall += results_total[:, i*num_values:(i+1)*num_values]
        results_total_overall /= num_set
        results_total_overall = torch.cat([results_total[:, :-2], results_total_overall, results_total[:, -2:]], dim=-1)

        for r in range(results_total_overall.shape[0]):
            data[f"run {r}"] = {}
            for n, ood_te in enumerate(ood_te_set):
                data[f"run {r}"][ood_te] = {}
                for i, t in enumerate(title[:-1]):
                    data[f"run {r}"][ood_te][t] = results_total_overall[r, n*num_values+i].item()
            data[f"run {r}"]["overall"] = {}
            for i, t in enumerate(title):
                data[f"run {r}"]["overall"][t] = results_total_overall[r, num_set*num_values+i].item()

        results_mean = results_total_overall.mean(dim=0)
        data["final"] = {}
        for n, ood_te in enumerate(ood_te_set):
            data["final"][ood_te] = {}
            for i, t in enumerate(title[:-1]):
                data["final"][ood_te][t] = results_mean[n*num_values+i].item()
        data["final"]["overall"] = {}
        for i, t in enumerate(title):
            data["final"]["overall"][t] = results_mean[num_set*num_values+i].item()

    return data


def return_things(save_file):
    ret = save_file['results']['final']
    if 'overall' in ret.keys():
        ret = ret['overall']

    return list(ret.values())


def save_file_data_epoch(ds, results_total, config, subtitle='epoch'):
    data = {}
    title = c.title[config['mode']]
    results_total = torch.tensor(results_total)

    if config['mode'] == 'classify' or (config['mode'] == 'detect' and ds not in ['twitch', 'arxiv']):
        for r in range(results_total.shape[0]):
            data[f"{subtitle} {r}"] = {}
            for i, t in enumerate(title):
                data[f"{subtitle} {r}"][t] = results_total[r, i].item()

    elif config['mode'] == 'detect' and ds in ['twitch', 'arxiv']:

        ood_te_set = c.test_sg if ds == 'twitch' else c.test_year
        num_values = len(title) - 1
        num_set = len(ood_te_set)
        results_total_overall = torch.zeros([results_total.shape[0], num_values])
        for i in range(num_set):
            results_total_overall += results_total[:, i*num_values:(i+1)*num_values]
        results_total_overall /= num_set
        results_total_overall = torch.cat([results_total[:, :-2], results_total_overall, results_total[:, -2:]], dim=-1)

        for r in range(results_total_overall.shape[0]):
            data[f"{subtitle} {r}"] = {}
            for n, ood_te in enumerate(ood_te_set):
                data[f"{subtitle} {r}"][ood_te] = {}
                for i, t in enumerate(title[:-1]):
                    data[f"{subtitle} {r}"][ood_te][t] = results_total_overall[r, n*num_values+i].item()
            data[f"{subtitle} {r}"]["overall"] = {}
            for i, t in enumerate(title):
                data[f"{subtitle} {r}"]["overall"][t] = results_total_overall[r, num_set*num_values+i].item()

    return data