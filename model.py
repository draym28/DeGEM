import torch
import torch.nn as nn
import torch.nn.functional as F
from dgi import DGI



class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, c, rho=1.):
        if rho != 0.:
            x = torch.cat([x, (rho * c).unsqueeze(0).expand_as(x)], dim=-1)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class DiscEnergyNet(nn.Module):
    def __init__(self, hidden_channels):
        super(DiscEnergyNet, self).__init__()
        self.proj = nn.Linear(hidden_channels, hidden_channels)

    def reset_parameters(self):
        self.proj.reset_parameters()

    def forward(self, x, c, rho=None):
        sc = -(self.proj(x) * c).sum(dim=-1)
        return sc


class DeGEM(nn.Module):
    def __init__(self, in_channels, num_classes, config):
        super(DeGEM, self).__init__()

        self.config = config
        self.rho = config['rho']
        self.gamma = config['gamma']

        # model for contrastive learning
        self.encoder = DGI(in_channels, config)

        # energy_net
        indim = config['cl_hdim']
        hdim = config['hidden_channels']
        outdim = 1
        if config['backbone'] == 'mlp':
            if self.rho == 0.:
                self.energy_net = MLP(indim, hdim, outdim, config['num_layers'], config['dropout'])
            else:
                self.energy_net = MLP(2*indim, hdim, outdim, config['num_layers'], config['dropout'])
        elif config['backbone'] == 'bilinear':
            self.energy_net = DiscEnergyNet(indim)

        # classifier
        self.classifier = nn.Linear(indim, num_classes)

        self.p = None
        self.c = None

        # hyper-params
        self.D = config['cl_hdim']  # dimension of MCMC sample data
        self.coef_reg = config['coef_reg']
        self.mcmc_steps = config['mcmc_steps']  # number of MCMC sampling
        self.mcmc_step_size = config['mcmc_step_size']  # grad coef of MCMC sampling
        self.mcmc_noise = config['mcmc_noise']  # noise of MCMC sampling

        self.max_buffer_vol = config['max_buffer_vol']
        self.buffer_prob = 0.95
        self.replay_buffer = None

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.energy_net.reset_parameters()

    def energy_readout(self, h, p, gamma=1.):
        if p is None:
            return h.mean(dim=0)
        elif gamma == 0:
            return h.mean(dim=0)
        elif gamma == 1:
            return (h * p.view(-1, 1)).sum(dim=0)
        else:
            return gamma * (h * p.view(-1, 1)).sum(dim=0) + (1 - gamma) * h.mean(dim=0)

    def forward(self, dataset):
        z_pos, z_neg = self.encoder(dataset)  # (n, cl_hdim), (n, cl_hdim)
        self.c = self.energy_readout(z_pos.detach(), self.p, self.gamma)  # (cl_hdim)
        energy = self.energy_net(z_pos.detach(), self.c, self.rho).squeeze()  # (n)
        return z_pos, z_neg, energy

    def classify(self, dataset, node_idx):
        z = self.encoder.embed(dataset)  # (n, cl_hdim)
        logits = self.classifier(z[node_idx].detach())
        return logits

    @torch.no_grad()
    def detect(self, dataset, node_idx):
        """return negative energy"""
        z = self.encoder.embed(dataset)  # (n, cl_hdim)
        neg_energy = -self.energy_net(z[node_idx], self.c, self.rho).squeeze()  # (n')

        return neg_energy

    def loss_compute(self, dataset_ind, dataset_ood=None, reduction='mean'):
        train_idx = dataset_ind.splits['train']
        edge_index, y = dataset_ind.edge_index, dataset_ind.y
        # z: (n, cl_hdim), energy: (n)
        embed_pos, embed_neg, energy_ind = self.forward(dataset_ind)

        # classify loss
        logits = self.classifier(embed_pos[train_idx])
        classify_loss = F.cross_entropy(logits, y[train_idx])

        # gen loss for data_ind
        num_nodes_sample = train_idx.shape[0]
        z_sample = self.sample(
            sample_size=num_nodes_sample,
            max_buffer_len=self.max_buffer_vol * num_nodes_sample,
            device=dataset_ind.x.device)  # (n', cl_hdim)

        energy_sample = self.energy_net(z_sample, self.c, self.rho).squeeze()  # (n')

        loss = self.gen_loss(energy_ind[train_idx], energy_sample, reduction)

        # update p & c
        self.p = torch.softmax(-energy_ind, dim=0)  # (n)
        self.c = self.energy_readout(embed_pos, self.p, self.gamma)  # (cl_hdim)
        self.p = self.p.detach()

        # cl loss
        loss = loss + self.encoder.loss_compute(embed_pos, embed_neg, self.c) \
               + self.config['lam'] * classify_loss

        return loss

    def gen_loss(self, energy_pos, energy_neg, reduction='mean'):
        # energy: (n)
        loss = energy_pos - energy_neg
        loss_reg = energy_pos.pow(2) + energy_neg.pow(2)
        loss = loss + self.coef_reg * loss_reg

        if reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'none' or reduction is None:
            return loss
        else:
            raise NotImplementedError

    def energy_gradient(self, x):
        self.encoder.eval()
        self.energy_net.eval()

        # copy data
        x_i = x.clone().detach()
        x_i.requires_grad = True  # need to compute gradient when sampling

        # compute gradient
        p_x = -self.energy_net(x_i, self.c, self.rho)  # (n)
        x_i_grad = torch.autograd.grad(p_x.sum(), [x_i], retain_graph=True)[0]

        self.encoder.train()
        self.energy_net.train()

        return x_i_grad

    def langevine_dynamics_step(self, x_old):
        # gradient wrt x_old
        grad_energy = self.energy_gradient(x_old)
        # sample eta ~ Normal(0, alpha)
        epsilon = torch.randn_like(grad_energy).to(x_old.device) * self.mcmc_noise

        # new sample
        x_new = x_old + self.mcmc_step_size * grad_energy + epsilon

        return x_new.detach()

    def sample(self, sample_size, max_buffer_len, device):
        # 1) init sample
        if self.replay_buffer is None:
            x_sample = 2. * torch.rand([sample_size, self.D]).to(device) - 1.
        else:
            replay_sample_idx = torch.randperm(self.replay_buffer.shape[0])[:int(sample_size * self.buffer_prob)]
            x_sample = torch.cat([self.replay_buffer[replay_sample_idx],
                                  torch.rand([sample_size - replay_sample_idx.shape[0], self.D]).to(device)], dim=0)

        # 2) run Langevine Dynamics sampling
        for _ in range(self.mcmc_steps):
            x_sample = self.langevine_dynamics_step(x_sample)
        if self.replay_buffer is None:
            self.replay_buffer = x_sample
        else:
            self.replay_buffer = torch.cat([x_sample, self.replay_buffer], dim=0)[:max_buffer_len]

        return x_sample
