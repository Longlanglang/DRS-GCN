import torch, random, os
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch.nn as nn
from random_sparse import random_sparse_layer
import torch.nn.functional as F
import time
from edge_curvature import GetEdgeWeight, WeightDrop
import config
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim, num_layers, seed,
                 dropout, using_sampling, dynamic_organization, withbn):
        super(GCN, self).__init__()
        torch.manual_seed(seed)
        self.dynamic_organization = dynamic_organization
        self.using_sampling = using_sampling
        self.num_layers = num_layers
        self.dropout = dropout
        self.withbn = withbn
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_dim)
        if num_layers > 2:
            self.convs = nn.ModuleList(GCNConv(hidden_channels, hidden_channels) for i in range(num_layers-2))
        if using_sampling:
            self.sampler_layer = random_sparse_layer
        if withbn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d(hidden_channels)
            if num_layers > 2:
                self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_channels) for i in range(num_layers - 2))

    def forward(self, x, edge_index, percent, ricci_curvature):
        res = []

        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)

        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn1(x)
        x = self.conv1(x, sampled_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        res.append(x)
        if self.num_layers > 2:
            for i in range(len(self.convs)):
                if self.using_sampling and self.training:
                    sampled_edge_index = self.sampler_layer(edge_index, percent)
                elif self.dynamic_organization and self.training:
                    weight = GetEdgeWeight(x, ricci_curvature, edge_index)
                    edge_weight = weight.compute_edge_weight()
                    weight_droper = WeightDrop(edge_index, edge_weight)
                    sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
                else:
                    sampled_edge_index = edge_index
                if self.withbn:
                    x = self.bns[i](x)
                x = self.convs[i](x, sampled_edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                res.append(x)

        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn2(x)
        x = self.conv2(x, sampled_edge_index)
        return x, res

class GCN1(torch.nn.Module):
    # add
    def __init__(self, input_dim, hidden_channels, output_dim, num_layers, seed,
                 dropout, using_sampling, dynamic_organization, withbn):
        super(GCN1, self).__init__()
        torch.manual_seed(seed)
        self.dynamic_organization = dynamic_organization
        self.using_sampling = using_sampling
        self.num_layers = num_layers
        self.dropout = dropout
        self.withbn = withbn
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_dim)
        if num_layers > 2:
            self.convs = nn.ModuleList(GCNConv(hidden_channels, hidden_channels) for i in range(num_layers-2))
        if using_sampling:
            self.sampler_layer = random_sparse_layer
        if withbn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d(hidden_channels)
            if num_layers > 2:
                self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_channels) for i in range(num_layers - 2))

    def forward(self, x, edge_index, percent, ricci_curvature):
        res = []

        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)

        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn1(x)
        x = self.conv1(x, sampled_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        res.append(x)
        if self.num_layers > 2:
            for i in range(len(self.convs)):
                if self.using_sampling and self.training:
                    sampled_edge_index = self.sampler_layer(edge_index, percent)
                elif self.dynamic_organization and self.training:
                    weight = GetEdgeWeight(x, ricci_curvature, edge_index)
                    edge_weight = weight.compute_edge_weight()
                    weight_droper = WeightDrop(edge_index, edge_weight)
                    sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
                else:
                    sampled_edge_index = edge_index
                if self.withbn:
                    x = self.bns[i](x)
                x = self.convs[i](x, sampled_edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                res.append(x)
                x += sum(res[:-1]) / (len(res) - 1)

        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn2(x)
        x = self.conv2(x, sampled_edge_index)
        return x, res

class GCN2(torch.nn.Module):
    # concat
    def __init__(self, input_dim, hidden_channels, output_dim, num_layers, seed,
                 dropout, using_sampling, dynamic_organization, withbn):
        super(GCN2, self).__init__()
        torch.manual_seed(seed)
        self.dynamic_organization = dynamic_organization
        self.using_sampling = using_sampling
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels * (num_layers - 1), output_dim)
        if num_layers > 2:
            self.convs = nn.ModuleList()
            for i in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels * (i + 1), hidden_channels))
        if using_sampling:
            self.sampler_layer = random_sparse_layer
        self.withbn = withbn
        if withbn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d(hidden_channels * (num_layers - 1))
            if num_layers > 2:
                self.bns = nn.ModuleList()
                for i in range(num_layers - 2):
                    self.bns.append(nn.BatchNorm1d(hidden_channels * (i + 1)))

    def forward(self, x, edge_index, percent, ricci_curvature):
        res = []

        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)

        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn1(x)
        x = self.conv1(x, sampled_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        res.append(x)
        if self.num_layers > 2:
            for i in range(len(self.convs)):
                if self.using_sampling and self.training:
                    sampled_edge_index = self.sampler_layer(edge_index, percent)
                elif self.dynamic_organization and self.training:
                    weight = GetEdgeWeight(x, ricci_curvature, edge_index)
                    edge_weight = weight.compute_edge_weight()
                    weight_droper = WeightDrop(edge_index, edge_weight)
                    sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
                else:
                    sampled_edge_index = edge_index
                if self.withbn:
                    x = self.bns[i](x)
                x = self.convs[i](x, sampled_edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                res.append(x)
                # x concat all output before current layer
                x = torch.cat(res, dim=1)
        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn2(x)
        x = self.conv2(x, sampled_edge_index)
        return x, res


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim, num_layers, seed,
                 dropout, using_sampling, dynamic_organization, withbn):
        super(GAT, self).__init__()
        torch.manual_seed(seed)
        self.dynamic_organization = dynamic_organization
        self.using_sampling = using_sampling
        self.num_layers = num_layers
        self.dropout = dropout
        self.negative_slope = 0.2
        self.concat = False
        self.conv1 = GATConv(in_channels=input_dim, out_channels=hidden_channels, heads=8,
                             concat=self.concat, negative_slope=self.negative_slope, dropout=self.dropout)
        self.conv2 = GATConv(in_channels=hidden_channels, out_channels=output_dim, heads=1,
                             concat=self.concat, negative_slope=self.negative_slope, dropout=self.dropout)
        if num_layers > 2:
            self.convs = nn.ModuleList(GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=8,
                                               concat=self.concat, negative_slope=self.negative_slope,
                                               dropout=self.dropout) for i in range(num_layers-2))
        if using_sampling:
            self.sampler_layer = random_sparse_layer
        self.withbn = withbn
        if withbn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d(hidden_channels)
            if num_layers > 2:
                self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_channels) for i in range(num_layers - 2))


    def forward(self, x, edge_index, percent, ricci_curvature):
        res = []

        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn1(x)
        x = self.conv1(x, sampled_edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        res.append(x)
        if self.num_layers > 2:
            for i in range(len(self.convs)):
                if self.using_sampling and self.training:
                    sampled_edge_index = self.sampler_layer(edge_index, percent)
                elif self.dynamic_organization and self.training:
                    weight = GetEdgeWeight(x, ricci_curvature, edge_index)
                    edge_weight = weight.compute_edge_weight()
                    weight_droper = WeightDrop(edge_index, edge_weight)
                    sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
                else:
                    sampled_edge_index = edge_index
                if self.withbn:
                    x = self.bns[i](x)
                x = self.convs[i](x, sampled_edge_index)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                res.append(x)
        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn2(x)
        x = self.conv2(x, sampled_edge_index)
        return x, res

class JKNet(nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim, num_layers, seed,
                 dropout, using_sampling, dynamic_organization, withbn):
        super(JKNet, self).__init__()
        torch.manual_seed(seed)
        self.dropout = dropout
        self.dynamic_organization = dynamic_organization
        self.using_sampling = using_sampling
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.convs = nn.ModuleList(GCNConv(hidden_channels, hidden_channels) for i in range(num_layers - 2))
        self.out_conv = GCNConv((num_layers-1) * hidden_channels, output_dim)
        self.withbn = withbn
        if withbn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d((num_layers-1) * hidden_channels)
            if num_layers > 2:
                self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_channels) for i in range(num_layers - 2))

    def forward(self, x, edge_index, percent, ricci_curvature):
        res = []
        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn1(x)
        x_i = self.conv1(x, sampled_edge_index)
        x_i = F.relu(x_i)
        x_i = F.dropout(x_i, p=self.dropout, training=self.training)
        res.append(x_i)
        for i in range(len(self.convs)):

            if self.using_sampling and self.training:
                sampled_edge_index = self.sampler_layer(edge_index, percent)
            elif self.dynamic_organization and self.training:
                weight = GetEdgeWeight(x_i, ricci_curvature, edge_index)
                edge_weight = weight.compute_edge_weight()
                weight_droper = WeightDrop(edge_index, edge_weight)
                sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
            else:
                sampled_edge_index = edge_index
            if self.withbn:
                x = self.bns[i](x_i)
            x_i = self.convs[i](x_i, sampled_edge_index)
            x_i = F.relu(x_i)
            x_i = F.dropout(x_i, p=self.dropout, training=self.training)
            res.append(x_i)

        x_cat = torch.cat(res, dim=1)

        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x_cat = self.bn2(x_cat)
        x = self.out_conv(x_cat, sampled_edge_index)
        return x, res

class IncepGCNBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_branch_layers, dropout, using_sampling, dynamic_organization, withbn):
        super(IncepGCNBranch, self).__init__()
        self.num_branch_layers = num_branch_layers
        self.using_sampling = using_sampling
        self.dynamic_organization = dynamic_organization
        self.dropout = dropout
        self.conv_head = GCNConv(in_channels, hidden_channels)
        if num_branch_layers > 1:
            self.conv_hidden = nn.ModuleList(GCNConv(hidden_channels, hidden_channels) for i in range(num_branch_layers-1))
        self.withbn = withbn
        if withbn:
            self.bn1 = nn.BatchNorm1d(in_channels)
            if num_branch_layers > 1:
                self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_channels) for i in range(num_branch_layers - 1))
    def forward(self, x, edge_index, percent, ricci_curvature):
        res_branch = []
        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn1(x)
        x = self.conv_head(x, sampled_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        res_branch.append(x)
        if self.num_branch_layers > 1:
            for i in range(len(self.conv_hidden)):
                if self.using_sampling and self.training:
                    sampled_edge_index = self.sampler_layer(edge_index, percent)
                elif self.dynamic_organization and self.training:
                    weight = GetEdgeWeight(x, ricci_curvature, edge_index)
                    edge_weight = weight.compute_edge_weight()
                    weight_droper = WeightDrop(edge_index, edge_weight)
                    sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
                else:
                    sampled_edge_index = edge_index
                if self.withbn:
                    x = self.bns[i](x)
                x = self.conv_hidden[i](x, sampled_edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                res_branch.append(x)
        return x, res_branch

class IncepGCN(nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim, num_layers, seed,
                 dropout, using_sampling, dynamic_organization, withbn):
        super(IncepGCN, self).__init__()
        torch.manual_seed(seed)
        self.dynamic_organization = dynamic_organization
        self.using_sampling = using_sampling
        self.num_layers = num_layers
        self.dropout = dropout
        self.split, self.branches = self.split_integer(num_layers-1)
        self.IncepGCNBranch = nn.ModuleList(IncepGCNBranch(input_dim, hidden_channels, self.split[i], dropout, using_sampling,
                                                           dynamic_organization, withbn) for i in range(self.branches))
        self.conv_out = GCNConv(hidden_channels * self.branches, output_dim)
        self.withbn = withbn
        if withbn:
            self.bn2 = nn.BatchNorm1d(hidden_channels * self.branches)



    def forward(self, x, edge_index, percent, ricci_curvature):
        res = []
        x_cat = []
        for Branch in self.IncepGCNBranch:
            x_branch, branch_res = Branch(x, edge_index, percent, ricci_curvature)
            res.append(branch_res)
            x_cat.append(x_branch)
        x_cat = torch.cat(x_cat, dim=1)
        flat_res = [item for sublist in res for item in sublist]
        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x_cat = self.bn2(x_cat)
        x = self.conv_out(x_cat, sampled_edge_index)
        return x, flat_res
    def split_integer(self, n):
        result = []
        current = 1
        while n > 0:
            if n >= current:
                result.append(current)
                n -= current
            else:
                result.append(n)
                break
            current += 1
        return result, len(result)


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim, num_layers, seed,
                 dropout, using_sampling, dynamic_organization, withbn):
        super(GraphSAGE, self).__init__()
        get_random_seed(seed)
        self.dynamic_organization = dynamic_organization
        self.using_sampling = using_sampling
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv1 = SAGEConv(input_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, output_dim)
        if num_layers > 2:
            self.convs = nn.ModuleList(SAGEConv(hidden_channels, hidden_channels) for i in range(num_layers-2))
        if using_sampling:
            self.sampler_layer = random_sparse_layer
        self.withbn = withbn
        if withbn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d(hidden_channels)
            if num_layers > 2:
                self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_channels) for i in range(num_layers - 2))


    def forward(self, x, edge_index, percent, ricci_curvature):
        res = []

        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn1(x)
        x = self.conv1(x, sampled_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        res.append(x)
        if self.num_layers > 2:
            for i in range(len(self.convs)):
                if self.using_sampling and self.training:
                    sampled_edge_index = self.sampler_layer(edge_index, percent)
                elif self.dynamic_organization and self.training:
                    weight = GetEdgeWeight(x, ricci_curvature, edge_index)
                    edge_weight = weight.compute_edge_weight()
                    weight_droper = WeightDrop(edge_index, edge_weight)
                    sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
                else:
                    sampled_edge_index = edge_index
                if self.withbn:
                    x = self.bns[i](x)
                x = self.convs[i](x, sampled_edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                res.append(x)
        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn2(x)
        x = self.conv2(x, sampled_edge_index)
        return x, res
class ResGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim, num_layers, seed,
                 dropout, using_sampling, dynamic_organization, withbn):
        super(ResGCN, self).__init__()
        get_random_seed(seed)
        self.dynamic_organization = dynamic_organization
        self.using_sampling = using_sampling
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(input_dim+hidden_channels, hidden_channels)
        self.conv_out = GCNConv(hidden_channels*2, output_dim)
        if num_layers > 3:
            self.convs = nn.ModuleList(GCNConv(hidden_channels*2, hidden_channels) for i in range(num_layers-3))
        if using_sampling:
            self.sampler_layer = random_sparse_layer
        self.withbn = withbn
        if withbn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d(hidden_channels+input_dim)
            self.bn_out = nn.BatchNorm1d(hidden_channels*2)
            if num_layers > 3:
                self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_channels*2) for i in range(num_layers - 3))

    def forward(self, x, edge_index, percent, ricci_curvature):
        res = []
        x_input = x
        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn1(x)
        x = self.conv1(x, sampled_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        res.append(x)
        x = torch.concat([x, x_input], dim=1)
        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn2(x)
        x = self.conv2(x, sampled_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        res.append(x)
        x = torch.concat([x, res[-2]], dim=1)
        if self.num_layers > 3:
            for i in range(len(self.convs)):
                if self.using_sampling and self.training:
                    sampled_edge_index = self.sampler_layer(edge_index, percent)
                elif self.dynamic_organization and self.training:
                    weight = GetEdgeWeight(x, ricci_curvature, edge_index)
                    edge_weight = weight.compute_edge_weight()
                    weight_droper = WeightDrop(edge_index, edge_weight)
                    sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
                else:
                    sampled_edge_index = edge_index
                if self.withbn:
                    x = self.bns[i](x)
                x = self.convs[i](x, sampled_edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                res.append(x)
                x = torch.concat([x, res[-2]], dim=1)
        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn_out(x)
        x = self.conv_out(x, sampled_edge_index)
        return x, res

class DenseGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim, num_layers, seed,
                 dropout, using_sampling, dynamic_organization, withbn):
        super(DenseGCN, self).__init__()
        torch.manual_seed(seed)
        self.dynamic_organization = dynamic_organization
        self.using_sampling = using_sampling
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels*(num_layers-1), output_dim)
        if num_layers > 2:
            self.convs = nn.ModuleList()
            for i in range(num_layers-2):
                self.convs.append(GCNConv(hidden_channels*(i+1), hidden_channels))
        if using_sampling:
            self.sampler_layer = random_sparse_layer
        self.withbn = withbn
        if withbn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d(hidden_channels*(num_layers-1))
            if num_layers > 2:
                self.bns = nn.ModuleList()
                for i in range(num_layers-2):
                    self.bns.append(nn.BatchNorm1d(hidden_channels*(i+1)))

    def forward(self, x, edge_index, percent, ricci_curvature):
        res = []

        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)

        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn1(x)
        x = self.conv1(x, sampled_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        res.append(x)
        if self.num_layers > 2:
            for i in range(len(self.convs)):
                if self.using_sampling and self.training:
                    sampled_edge_index = self.sampler_layer(edge_index, percent)
                elif self.dynamic_organization and self.training:
                    weight = GetEdgeWeight(x, ricci_curvature, edge_index)
                    edge_weight = weight.compute_edge_weight()
                    weight_droper = WeightDrop(edge_index, edge_weight)
                    sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
                else:
                    sampled_edge_index = edge_index
                if self.withbn:
                    x = self.bns[i](x)
                x = self.convs[i](x, sampled_edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                res.append(x)
                x = torch.cat(res, dim=1)
        if self.using_sampling and self.training:
            sampled_edge_index = self.sampler_layer(edge_index, percent)
        elif self.dynamic_organization and self.training:
            weight = GetEdgeWeight(x, ricci_curvature, edge_index)
            edge_weight = weight.compute_edge_weight()
            weight_droper = WeightDrop(edge_index, edge_weight)
            sampled_edge_index, edge_weight = weight_droper.weight_drop(percent)
        else:
            sampled_edge_index = edge_index
        if self.withbn:
            x = self.bn2(x)
        x = self.conv2(x, sampled_edge_index)
        return x, res
