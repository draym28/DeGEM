import numpy as np
import scipy.sparse as sp
from termcolor import cprint

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, AttributedGraphDataset, WikipediaNetwork, WebKB, Actor, Reddit
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, add_self_loops, degree, homophily
from torch_sparse import SparseTensor

from model import DeGEM
import config as c


device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
ds_list = [
    'twitch', 'arxiv', 
    'cora', 'citeseer', 'pubmed', 
    'amazon-photo', 'amazon-computer', 'coauthor-cs', 'coauthor-physics', 
    'wiki', 'blogcatalog', 
    'chameleon', 'squirrel', 
    'actor', 
    'texas', 'cornell', 'wisconsin', 
    # 'ogbn-papers100M', 
    'reddit', 
]


def print_dataset_info(data_dir):
    transform = T.NormalizeFeatures()
    for dataname in ds_list:
        if dataname == 'twitch':
            subgraph_names = ['DE', 'EN', 'ES', 'FR', 'RU']
            for i in range(len(subgraph_names)):
                dataset = Twitch(root=f'{data_dir}Twitch', name=subgraph_names[i], transform=transform)[0]
                cprint(f'{dataname} {subgraph_names[i]}', 'green')
                print(dataset)
                node_homo = homophily(dataset.edge_index, dataset.y, method='node')
                cprint(f'num classes: {dataset.y.unique().shape[0]}, node homophily: {node_homo}', 'yellow')

        elif dataname == 'arxiv':
            from ogb.nodeproppred import NodePropPredDataset
            dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')[0]
            cprint(f'{dataname}', 'green')
            print(dataset)

        elif dataname in ('cora', 'citeseer', 'pubmed'):
            torch_dataset = Planetoid(root=f'{data_dir}Planetoid', split='public',
                                name=dataname, transform=transform)
            dataset = torch_dataset[0]
            cprint(f'{dataname}', 'green')
            print(dataset)
            node_homo = homophily(dataset.edge_index, dataset.y, method='node')
            cprint(f'num classes: {dataset.y.unique().shape[0]}, node homophily: {node_homo}', 'yellow')

        elif dataname == 'amazon-photo':
            torch_dataset = Amazon(root=f'{data_dir}Amazon', name='Photo', transform=transform)
            dataset = torch_dataset[0]
            cprint(f'{dataname}', 'green')
            print(dataset)
            node_homo = homophily(dataset.edge_index, dataset.y, method='node')
            cprint(f'num classes: {dataset.y.unique().shape[0]}, node homophily: {node_homo}', 'yellow')

        elif dataname == 'amazon-computer':
            torch_dataset = Amazon(root=f'{data_dir}Amazon', name='Computers', transform=transform)
            dataset = torch_dataset[0]
            cprint(f'{dataname}', 'green')
            print(dataset)
            node_homo = homophily(dataset.edge_index, dataset.y, method='node')
            cprint(f'num classes: {dataset.y.unique().shape[0]}, node homophily: {node_homo}', 'yellow')

        elif dataname == 'coauthor-cs':
            torch_dataset = Coauthor(root=f'{data_dir}Coauthor', name='CS', transform=transform)
            dataset = torch_dataset[0]
            cprint(f'{dataname}', 'green')
            print(dataset)
            node_homo = homophily(dataset.edge_index, dataset.y, method='node')
            cprint(f'num classes: {dataset.y.unique().shape[0]}, node homophily: {node_homo}', 'yellow')

        elif dataname == 'coauthor-physics':
            torch_dataset = Coauthor(root=f'{data_dir}Coauthor', name='Physics', transform=transform)
            dataset = torch_dataset[0]
            cprint(f'{dataname}', 'green')
            print(dataset)
            node_homo = homophily(dataset.edge_index, dataset.y, method='node')
            cprint(f'num classes: {dataset.y.unique().shape[0]}, node homophily: {node_homo}', 'yellow')

        elif dataname in ['wiki', 'blogcatalog']:
            torch_dataset = AttributedGraphDataset(root=f'{data_dir}AttributedGraphDataset', name=dataname, transform=transform)
            dataset = torch_dataset[0]
            cprint(f'{dataname}', 'green')
            print(dataset)
            node_homo = homophily(dataset.edge_index, dataset.y, method='node')
            cprint(f'num classes: {dataset.y.unique().shape[0]}, node homophily: {node_homo}', 'yellow')

        elif dataname in ['chameleon', 'squirrel']:
            torch_dataset = WikipediaNetwork(root=f'{data_dir}WikipediaNetwork', name=dataname, transform=transform)
            dataset = torch_dataset[0]
            cprint(f'{dataname}', 'green')
            print(dataset)
            node_homo = homophily(dataset.edge_index, dataset.y, method='node')
            cprint(f'num classes: {dataset.y.unique().shape[0]}, node homophily: {node_homo}', 'yellow')

        elif dataname in ['texas', 'cornell', 'wisconsin']:
            torch_dataset = WebKB(root=f'{data_dir}WebKB', name=dataname, transform=transform)
            dataset = torch_dataset[0]
            cprint(f'{dataname}', 'green')
            print(dataset)
            node_homo = homophily(dataset.edge_index, dataset.y, method='node')
            cprint(f'num classes: {dataset.y.unique().shape[0]}, node homophily: {node_homo}', 'yellow')

        elif dataname == 'actor':
            torch_dataset = Actor(root=f'{data_dir}Actor', transform=transform)
            dataset = torch_dataset[0]
            cprint(f'{dataname}', 'green')
            print(dataset)
            node_homo = homophily(dataset.edge_index, dataset.y, method='node')
            cprint(f'num classes: {dataset.y.unique().shape[0]}, node homophily: {node_homo}', 'yellow')

        elif dataname == 'ogbn-papers100M':
            from ogb.nodeproppred import NodePropPredDataset
            dataset = NodePropPredDataset(name='ogbn-papers100M', root=f'{data_dir}/ogb')[0]
            cprint(f'{dataname}', 'green')
            print(dataset)

        elif dataname == 'reddit':
            dataset = Reddit(root=f'{data_dir}/Reddit')[0]
            cprint(f'{dataname}', 'green')
            print(dataset)

        else:
            raise NotImplementedError


def load_dataset(dataname, data_dir, config):
    '''
    dataset_ind: in-distribution training dataset
    dataset_ood_tr: ood-distribution training dataset as ood exposure
    dataset_ood_te: a list of ood testing datasets or one ood testing dataset
    '''
    # multi-graph datasets, use one as ind, the other as ood
    if dataname == 'twitch':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_twitch_dataset(data_dir)

    # single graph, use partial nodes as ind, others as ood according to domain info
    elif dataname in 'arxiv':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_arxiv_dataset(data_dir)

    # single graph, use original as ind, modified graphs as ood
    elif dataname in ('cora', 'citeseer', 'pubmed', 'amazon-photo', 'amazon-computer', 'coauthor-cs', 'coauthor-physics', 'wiki', 'blogcatalog'):
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_graph_dataset(data_dir, dataname, config)

    elif dataname in ('chameleon', 'squirrel', 'actor', 'cornell', 'texas', 'wisconsin'):
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_graph_dataset(data_dir, dataname, config)

    elif dataname in ('reddit'):
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_graph_dataset(data_dir, dataname, config)

    else:
        raise ValueError('Invalid dataname')

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_dataset_pair(dataname_ind, dataname_ood, data_dir, config):
    '''
    dataset_ind: in-distribution training dataset
    dataset_ood: ood-distribution testing dataset
    '''
    # multi-graph datasets, use one as ind, the other as ood

    dataset_ind, _, _ = load_dataset(dataname_ind, data_dir, config)
    dataset_ood, _, _ = load_dataset(dataname_ood, data_dir, config)

    return dataset_ind, dataset_ood


def prepare_dataset(dataset, device, model:DeGEM, cuda=True):
    def _func(d):
        N = d.x.shape[0]
        from torch_geometric.utils import to_undirected
        edge_index = to_undirected(d.edge_index.to(device), num_nodes=N)
        row, col = edge_index
        deg = degree(col, N).float()
        if cuda:
            deg_norm = torch.pow(deg.cuda(), -0.5).to(device)
        else:
            deg_norm = torch.pow(deg.cpu(), -0.5).to(device)
        deg_norm = torch.nan_to_num(deg_norm, nan=0.0, posinf=0.0, neginf=0.0)
        value = deg_norm[col] * deg_norm[row]
        d.edge_index = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        d.x_prop = model.encoder.conv.prop(d.x, d.edge_index)

    if isinstance(dataset, list):
        for d in dataset:
            _func(d)
    else:
        _func(dataset)


def load_twitch_dataset(data_dir):
    transform = T.NormalizeFeatures()
    subgraph_names = ['DE', 'EN', 'ES', 'FR', 'RU']
    train_idx, valid_idx = 0, 1
    dataset_ood_te = []
    for i in range(len(subgraph_names)):
        torch_dataset = Twitch(root=f'{data_dir}Twitch', name=subgraph_names[i], transform=transform)
        dataset = torch_dataset[0]
        dataset.node_idx = torch.arange(dataset.num_nodes)
        if i == train_idx:
            dataset_ind = dataset
        elif i == valid_idx:
            dataset_ood_tr = dataset
        else:
            dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_arxiv_dataset(data_dir, time_bound=[2015,2017], inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    year = ogb_dataset.graph['node_year']

    year_min, year_max = time_bound[0], time_bound[1]
    test_year_bound = [2017, 2018, 2019, 2020]

    idx = torch.arange(label.size(0))

    # ind nodes, for training
    center_node_mask = (year <= year_min).squeeze(1)
    if inductive:
        ind_edge_index, _ = subgraph(idx[center_node_mask], edge_index)
    else:
        ind_edge_index = edge_index
    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    dataset_ind.node_idx = idx[center_node_mask]

    # ood nodes, for training
    center_node_mask = (year <= year_max).squeeze(1) * (year > year_min).squeeze(1)
    if inductive:
        all_node_mask = (year <= year_max).squeeze(1)
        ood_tr_edge_index, _ = subgraph(idx[all_node_mask], edge_index)
    else:
        ood_tr_edge_index = edge_index
    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    dataset_ood_tr.node_idx = idx[center_node_mask]

    # ood nodes, for test
    dataset_ood_te = []
    for i in range(len(test_year_bound)-1):
        center_node_mask = (year <= test_year_bound[i+1]).squeeze(1) * (year > test_year_bound[i]).squeeze(1)
        if inductive:
            all_node_mask = (year <= test_year_bound[i+1]).squeeze(1)
            ood_te_edge_index, _ = subgraph(idx[all_node_mask], edge_index)
        else:
            ood_te_edge_index = edge_index

        dataset = Data(x=node_feat, edge_index=ood_te_edge_index, y=label)
        dataset.node_idx = idx[center_node_mask]
        dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def create_sbm_dataset(data, p_ii=1.5, p_ij=0.5):
    n = data.num_nodes

    d = data.edge_index.size(1) / data.num_nodes / (data.num_nodes - 1)
    num_blocks = int(data.y.max()) + 1
    p_ii, p_ij = p_ii * d, p_ij * d
    block_size = n // num_blocks
    block_sizes = [block_size for _ in range(num_blocks-1)] + [block_size + n % block_size]
    edge_probs = torch.ones((num_blocks, num_blocks)) * p_ij
    edge_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = p_ii
    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)

    dataset = Data(x=data.x, edge_index=edge_index, y=data.y)
    dataset.node_idx = torch.arange(dataset.num_nodes)

    # if hasattr(data, 'train_mask'):
    #     tensor_split_idx = {}
    #     idx = torch.arange(data.num_nodes)
    #     tensor_split_idx['train'] = idx[data.train_mask]
    #     tensor_split_idx['valid'] = idx[data.val_mask]
    #     tensor_split_idx['test'] = idx[data.test_mask]
    #
    #     dataset.splits = tensor_split_idx

    return dataset


def create_feat_noise_dataset(data):

    x = data.x
    n = data.num_nodes
    idx = torch.randint(0, n, (n, 2))
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)

    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)

    # if hasattr(data, 'train_mask'):
    #     tensor_split_idx = {}
    #     idx = torch.arange(data.num_nodes)
    #     tensor_split_idx['train'] = idx[data.train_mask]
    #     tensor_split_idx['valid'] = idx[data.val_mask]
    #     tensor_split_idx['test'] = idx[data.test_mask]
    #
    #     dataset.splits = tensor_split_idx

    return dataset


def create_label_noise_dataset(data):

    y = data.y
    n = data.num_nodes
    idx = torch.randperm(n)[:int(n * 0.5)]
    y_new = y.clone()
    y_new[idx] = torch.randint(0, y.max(), (int(n * 0.5), ))

    dataset = Data(x=data.x, edge_index=data.edge_index, y=y_new)
    dataset.node_idx = torch.arange(n)

    # if hasattr(data, 'train_mask'):
    #     tensor_split_idx = {}
    #     idx = torch.arange(data.num_nodes)
    #     tensor_split_idx['train'] = idx[data.train_mask]
    #     tensor_split_idx['valid'] = idx[data.val_mask]
    #     tensor_split_idx['test'] = idx[data.test_mask]
    #
    #     dataset.splits = tensor_split_idx

    return dataset


def load_graph_dataset(data_dir, dataname, config):
    transform = T.NormalizeFeatures()
    if dataname in ('cora', 'citeseer', 'pubmed'):
        torch_dataset = Planetoid(root=f'{data_dir}Planetoid', split='public', name=dataname, transform=transform)
        dataset = torch_dataset[0]
        tensor_split_idx = {}
        idx = torch.arange(dataset.num_nodes)
        tensor_split_idx['train'] = idx[dataset.train_mask]
        tensor_split_idx['valid'] = idx[dataset.val_mask]
        tensor_split_idx['test'] = idx[dataset.test_mask]
        dataset.splits = tensor_split_idx
    elif dataname == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon', name='Photo', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon', name='Computers', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor', name='CS', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor', name='Physics', transform=transform)
        dataset = torch_dataset[0]
    elif dataname in ['wiki', 'blogcatalog']:
        torch_dataset = AttributedGraphDataset(root=f'{data_dir}AttributedGraphDataset', name=dataname, transform=transform)
        dataset = torch_dataset[0]
    elif dataname in ['chameleon', 'squirrel']:
        torch_dataset = WikipediaNetwork(root=f'{data_dir}WikipediaNetwork', name=dataname, transform=transform)
        dataset = torch_dataset[0]
    elif dataname in ['texas', 'cornell', 'wisconsin']:
        torch_dataset = WebKB(root=f'{data_dir}WebKB', name=dataname, transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'actor':
        torch_dataset = Actor(root=f'{data_dir}Actor', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'reddit':
        torch_dataset = Reddit(root=f'{data_dir}/Reddit', transform=transform)
        dataset = torch_dataset[0]
    else:
        raise NotImplementedError

    dataset.node_idx = torch.arange(dataset.num_nodes)
    dataset_ind = dataset

    if config['ood_type'] == 'structure':
        dataset_ood_tr = create_sbm_dataset(dataset, p_ii=1.5, p_ij=0.5)
        dataset_ood_te = create_sbm_dataset(dataset, p_ii=1.5, p_ij=0.5)

    elif config['ood_type'] == 'feature':
        dataset_ood_tr = create_feat_noise_dataset(dataset)
        dataset_ood_te = create_feat_noise_dataset(dataset)

    elif config['ood_type'] in ['label', 'label-sep']:
        class_t_dict = {
            'cora': 3, 
            'amazon-photo': 4, 
            'chameleon': 1,  # 5
            'actor': 1,  # 5
            'cornell': 1,  # 5
        }
        class_t = class_t_dict[dataname]
        label = dataset.y

        center_node_mask_ind = (label > class_t)
        idx = torch.arange(label.size(0))
        dataset_ind.node_idx = idx[center_node_mask_ind]

        if dataname in ('cora', 'citeseer', 'pubmed'):
            split_idx = dataset.splits
            tensor_split_idx = {}
            idx = torch.arange(label.size(0))
            for key in split_idx:
                mask = torch.zeros(label.size(0), dtype=torch.bool)
                mask[torch.as_tensor(split_idx[key])] = True
                tensor_split_idx[key] = idx[mask * center_node_mask_ind]
            dataset_ind.splits = tensor_split_idx

        dataset_ood_tr = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)
        dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)

        center_node_mask_ood_tr = (label == class_t)
        center_node_mask_ood_te = (label < class_t)
        dataset_ood_tr.node_idx = idx[center_node_mask_ood_tr]
        dataset_ood_te.node_idx = idx[center_node_mask_ood_te]

        if config['ood_type'] == 'label-sep':
            # ID & OOD nodes have no edge linked
            idx = torch.arange(dataset.x.shape[0])
            dataset_ind.edge_index = subgraph(idx[dataset_ind.node_idx], dataset_ind.edge_index)[0]
            dataset_ood_tr.edge_index = subgraph(idx[dataset_ood_tr.node_idx], dataset_ood_tr.edge_index)[0]
            dataset_ood_te.edge_index = subgraph(idx[dataset_ood_te.node_idx], dataset_ood_te.edge_index)[0]

    else:
        raise NotImplementedError

    return dataset_ind, dataset_ood_tr, dataset_ood_te


if __name__ == "__main__":
    print_dataset_info(c.DATA_DIR)