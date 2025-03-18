import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MHEncoder


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.proj = nn.Linear(config['cl_hdim'], config['cl_hdim'])
        self.config = config
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.proj.reset_parameters()

    def sim(self, h, c):
        out = (self.proj(h) * c).sum(dim=-1)
        return out

    def forward(self, c, h_pl, h_mi):
        # c: (hdim), global readout
        # h_pl: (n, hdim)
        # h_mi: (n, hdim)

        c_x = torch.unsqueeze(c, 0)  # (1, hdim)
        c_x = c_x.expand_as(h_pl)    # (n, hdim)

        # D(h_i, s)
        sc_1 = self.sim(h_pl, c_x)  # (n)

        # D(h_j~, s)
        sc_2 = self.sim(h_mi, c_x)  # (n)

        logits = torch.cat((sc_1, sc_2), -1)  # (2n)

        return logits


class DGI(nn.Module):
    def __init__(self, in_channels, config):
        super(DGI, self).__init__()
        self.conv = MHEncoder(in_channels, config)
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(config)

    @torch.no_grad()
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.disc.reset_parameters()

    def forward(self, dataset):
        # representations of positive nodes
        h_1 = self.conv(dataset.x_prop)  # H: (n, hdim)

        # representations of negative nodes
        rand_idx = torch.randperm(dataset.x.shape[0])
        h_2 = self.conv(dataset.x[rand_idx], dataset.edge_index)  # H~: (n, hdim)

        return h_1, h_2

    def embed(self, dataset):
        h_1 = self.conv(dataset.x_prop)  # (n, hdim)
        return h_1

    def loss_compute(self, z_pos, z_neg, c):
        # z: (n, cl_hdim), px: (n)
        num_nodes = z_pos.shape[0]
        logits = self.disc(c, z_pos, z_neg)
        lbl_1 = torch.ones(num_nodes)   # (n)  positive pair, label = 1
        lbl_2 = torch.zeros(num_nodes)  # (n)  negative pair, label = 0
        lbl = torch.cat((lbl_1, lbl_2), dim=0).to(z_pos.device)  # (2n)
        return F.binary_cross_entropy_with_logits(logits, lbl)


class LogReg(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, x):
        out = self.fc(x)
        return out