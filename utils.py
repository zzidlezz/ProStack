import math
import os.path as osp
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import deeprobust.graph.utils as utils
import faiss

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets.amazon_products import AmazonProducts
from torch_geometric.datasets.reddit import Reddit
from torch_geometric.datasets.reddit2 import Reddit2
from torch_geometric.datasets.flickr import Flickr
from torch_geometric.datasets.s3dis import S3DIS

from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from deeprobust.graph.utils import *
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import add_remaining_self_loops, to_undirected, dropout_adj
from torch_geometric.loader import *
from torch_sparse import SparseTensor

def get_dataset(name, normalize_features=False, transform=None):
    if name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        root = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    else:
        root = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

    if name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        dataset = PygNodePropPredDataset(name=name, root=root, transform=T.ToSparseTensor())
    elif name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root, name)
    elif name=='amazon-products':
        dataset = AmazonProducts(root+'/'+name)
    elif name=='reddit':
        dataset = Reddit(root+'/'+name)
    elif name=='reddit2':
        dataset = Reddit2(root+'/'+name)
    elif name=='flickr':
        dataset = Flickr(root+'/'+name)
    else:
        raise NotImplementedError

    if transform is not None and normalize_features:#ogb不管用
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:#归一
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform


    dpr_data = Pyg2Dpr(dataset)
    if name in ['ogbn-arxiv', 'reddit2']:
        feat, idx_train = dpr_data.features, dpr_data.idx_train
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)
        dpr_data.features = feat

    return dpr_data


class Pyg2Dpr(Dataset):#input dataset and get the divided one. if we input partitioned dataset, then we can get what we want
    def __init__(self, pyg_data, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        try:
            dataset_name = pyg_data.name
        except:
            pass

        pyg_data = pyg_data.data


        try:
            if dataset_name == 'ogbn-papers100M':
                pyg_data.edge_index, _ = dropout_adj(pyg_data.edge_index, p = 0.4, num_nodes= pyg_data.num_nodes)
            if dataset_name in ['ogbn-arxiv', 'ogbn-papers100M']: 
                pyg_data.edge_index = to_undirected(edge_index=pyg_data.edge_index, edge_attr=None, num_nodes=pyg_data.num_nodes)
                a = pyg_data.edge_index[0]

        except:
            pass
        
        n = pyg_data.num_nodes
        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),  # 化为普通的稀疏矩阵
                                  (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))


        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()
        if self.labels.shape[-1]==107:
            self.labels = np.argmax(self.labels, -1)
        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape
            
        if hasattr(pyg_data, 'train_mask'):
            # for fixed split


            self.idx_train = mask_to_index(pyg_data.train_mask, n)
            self.idx_val = mask_to_index(pyg_data.val_mask, n)
            self.idx_test = mask_to_index(pyg_data.test_mask, n)


            self.train_mask = pyg_data.train_mask
            self.val_mask = pyg_data.val_mask
            self.test_mask = pyg_data.test_mask


            self.name = 'Pyg2Dpr'
        else:
            try:
                # for ogb
                self.idx_train = splits['train']#papers100M可能只选取了部分作为使用index
                self.idx_val = splits['valid']
                self.idx_test = splits['test']
                self.name = 'Pyg2Dpr'
            except:
                # for other datasets
                self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                        nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)
        print("train val test的长度:",len(self.idx_train),len(self.idx_val),len(self.idx_test))

class Transd2Ind:
    # transductive setting to inductive setting

    def __init__(self, dpr_data, keep_ratio):
        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        adj, features, labels = dpr_data.adj, dpr_data.features, dpr_data.labels
        self.nclass = labels.max()+1
        self.adj, self.features, self.labels = adj, features, labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)

        if keep_ratio < 1:
            idx_train, _ = train_test_split(idx_train,
                                            random_state=None,
                                            train_size=keep_ratio,
                                            test_size=1-keep_ratio,
                                            stratify=labels[idx_train])

        self.adj_train = adj[np.ix_(idx_train, idx_train)]
        self.adj_val = adj[np.ix_(idx_val, idx_val)]
        self.adj_test = adj[np.ix_(idx_test, idx_test)]
        print('size of adj_train:', self.adj_train.shape)
        print('#edges in adj_train:', self.adj_train.sum())

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]

        self.feat_train = features[idx_train]
        self.feat_val = features[idx_val]
        self.feat_test = features[idx_test]
        
import torch.sparse as ts
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import LSTM
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]



def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]


