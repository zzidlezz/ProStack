import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from models.mlp import MLP 
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm

class SGC_Multi(MessagePassing):


    _cached_x: Optional[Tensor]

    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout, K: int = 2, nlayers: int = 2, norm: Union[str, Callable, None] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.nfeat = nfeat
        self.nclass = nclass
        self.K = K
        # self.mlp = MLP(channel_list=[nfeat, 256, 256, nclass], nfeat=nfeat, nhid=256, nclass=nclass, dropout=[0.5, 0.5, 0.5], num_layers=3, norm='BatchNorm')
        channel_list=[nfeat]
        channel_list.extend([nhid]*(nlayers-1))
        channel_list.append(nclass)
        self.mlp = MLP(channel_list=channel_list, nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=[dropout]*nlayers, num_layers=nlayers, norm=norm)

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.initialize()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        for k in range(self.K):#只能用于full-batch，mini-batch显然连续propagate不合适
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                size=None)

        return self.mlp(x)

    def forward_sampler(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        for i, (adj, _, size) in enumerate(edge_index):
            x = self.propagate(adj, x=x, edge_weight=edge_weight)

        return self.mlp(x)
    
    @torch.no_grad()
    def predict(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        self.eval()
        return self.forward(x, edge_index, edge_weight=edge_weight)
    
    @torch.no_grad()
    def inference(self, x_all: Tensor, loader: NeighborSampler,
                  device: Optional[torch.device] = None,
                  progress_bar: bool = False) -> Tensor:
        self.eval()
        for i in range(self.K):
            xs: List[Tensor] = []
            for batch_size, n_id, adj in loader:
                x = x_all[n_id].to(device)#全局index
                edge_index = adj.adj_t.to(device)#只有SparseTensor而没有edge_index
                x = self.propagate(edge_index, x=x)[:batch_size]
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        x_all=self.mlp(x_all.to(device))
        return F.log_softmax(x_all,dim=1).to(device)
    
    
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.nfeat}, '
                f'{self.nclass}, K={self.K})')
