#########################################################
# This section of code has been adapted from kamilest/brain-age-gnn#
# Modified by Margarita Bintsi#
#########################################################

"""Implements parent BrainGNN and child BrainGCN, BrainGAT, GraphSAGE classes."""


import enum

import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, ChebConv,SAGEConv, GraphConv, GINConv, ChebConv
import torch.nn as nn
import torch.nn.functional as F

"""Implements parent BrainGNN and child BrainGCN, BrainGAT, GraphSAGE classes."""
class ConvTypes(enum.Enum):
    GCN = 'gcn'
    GAT = 'gat'
    SAGE = 'sage'
    GraphConv = 'graphconv'
    CHEB = 'cheb'

class BrainGNN(torch.nn.Module):
    def __init__(self, conv_type, num_node_features, n_conv_layers, layer_sizes, dropout_p):
        """
        Initialises BrainGNN class.
        :param conv_type: convolution type, either ConvType.GCN or ConvType.GAT; defaults to fully connected layers.
        :param num_node_features: number of input features.
        :param n_conv_layers: number of convolutional layers.
        :param layer_sizes: array of layer sizes, first of which convolutional, the rest fully connected.
        :param dropout_p: probability of dropping out a unit.
        """

        super(BrainGNN, self).__init__()
        
        self.conv = torch.nn.ModuleList()
        self.fc = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()
        self.relu = nn.ReLU()
        
        size = num_node_features
        self.params = torch.nn.ParameterList([size].extend(layer_sizes))
        for i in range(n_conv_layers):
            if conv_type == ConvTypes.GCN:
                self.conv.append(GCNConv(size, layer_sizes[i], improved=True))
            elif conv_type == ConvTypes.GAT:
                self.conv.append(GATConv(size, layer_sizes[i]))
            elif conv_type == ConvTypes.SAGE:
                self.conv.append(SAGEConv(size, layer_sizes[i]))
            elif conv_type == ConvTypes.GraphConv:
                self.conv.append(GraphConv(size, layer_sizes[i]))
            elif conv_type == ConvTypes.CHEB:
                self.conv.append(ChebConv(size, layer_sizes[i], 2))
            else:
                self.conv.append(Linear(size, layer_sizes[i]))
            size = layer_sizes[i]
        
        if (len(layer_sizes) - n_conv_layers) > 0:
            for i in range(len(layer_sizes) - n_conv_layers):
                self.fc.append(Linear(size, layer_sizes[n_conv_layers+i]))
                size = layer_sizes[n_conv_layers+i]
                if i < len(layer_sizes) - n_conv_layers - 1:
                    self.dropout.append(torch.nn.Dropout(dropout_p))

    def forward(self, x, edge_index):
        for i in range(len(self.conv)-1):
            x = self.conv[i](x, edge_index) 
            x = self.relu(x)
            
        x = self.conv[-1](x, edge_index)
        
        if (len(self.fc)) > 0:
            for i in range(len(self.fc) - 1):
                x = self.fc[i](x)
                x = self.relu(x)
                x = self.dropout[i](x)

            x = self.fc[-1](x)
        return x
    
class BrainGCN(BrainGNN):
    def __init__(self, num_node_features, n_conv_layers, layer_sizes, dropout_p):
        """
        Initialises GCN class.
        :param num_node_features: number of input features.
        :param n_conv_layers: number of convolutional layers.
        :param layer_sizes: array of layer sizes, first of which convolutional, the rest fully connected.
        :param dropout_p: probability of dropping out a unit.
        """
        super(BrainGCN, self).__init__(ConvTypes.GCN, num_node_features, n_conv_layers, layer_sizes, dropout_p)

class BrainGAT(BrainGNN):
    def __init__(self, num_node_features, n_conv_layers, layer_sizes, dropout_p):
        """
        Initialises BrainGNN class.
        :param num_node_features: number of input features.
        :param n_conv_layers: number of convolutional (attentional) layers.
        :param layer_sizes: array of layer sizes, first of which convolutional, the rest fully connected.
        :param dropout_p: probability of dropping out a unit.
        """
        super(BrainGAT, self).__init__(ConvTypes.GAT, num_node_features, n_conv_layers, layer_sizes, dropout_p)

class BrainSage(BrainGNN):
    def __init__(self, num_node_features, n_conv_layers, layer_sizes, dropout_p):
        """
        Initialises BrainGNN class.
        :param num_node_features: number of input features.
        :param n_conv_layers: number of convolutional (attentional) layers.
        :param layer_sizes: array of layer sizes, first of which convolutional, the rest fully connected.
        :param dropout_p: probability of dropping out a unit.
        """
        super(BrainSage, self).__init__(ConvTypes.SAGE, num_node_features, n_conv_layers, layer_sizes, dropout_p)

class BrainCheb(BrainGNN):
    def __init__(self, num_node_features, n_conv_layers, layer_sizes, dropout_p):
        """
        Initialises BrainGNN class.
        :param num_node_features: number of input features.
        :param n_conv_layers: number of convolutional (attentional) layers.
        :param layer_sizes: array of layer sizes, first of which convolutional, the rest fully connected.
        :param dropout_p: probability of dropping out a unit.
        """
        super(BrainCheb, self).__init__(ConvTypes.CHEB, num_node_features, n_conv_layers, layer_sizes, dropout_p)

class GraphConv(BrainGNN):
    def __init__(self, num_node_features, n_conv_layers, layer_sizes, dropout_p):
        """
        Initialises BrainGNN class.
        :param num_node_features: number of input features.
        :param n_conv_layers: number of convolutional (attentional) layers.
        :param layer_sizes: array of layer sizes, first of which convolutional, the rest fully connected.
        :param dropout_p: probability of dropping out a unit.
        """
        super(GraphConv, self).__init__(ConvTypes.GraphConv, num_node_features, n_conv_layers, layer_sizes, dropout_p)

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(68,512)
        self.linear2 = nn.Linear(512,128)
        self.linear3 = nn.Linear(128,1)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x 

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(68, 64),
                       BatchNorm1d(64), ReLU(),
                       Linear(64, 64), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(64, 64), BatchNorm1d(64), ReLU(),
                       Linear(64, 64), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(64, 64), BatchNorm1d(64), ReLU(),
                       Linear(64, 64), ReLU()))
        self.lin1 = Linear(64*3, 64*3)
        self.lin2 = Linear(64*3, 1)

    def forward(self, x, edge_index):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h
