import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.utils import spmm
from torch_sparse import SparseTensor


class DisenConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="mean", normalize=False, root_weight=True, bias=True):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_homo = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_hetero = Linear(in_channels[0], out_channels, bias=bias)

        if root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_homo.reset_parameters()
        self.lin_hetero.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: OptPairTensor, edge_index: Adj, size: Size = None):
        if isinstance(x, Tensor):
            x = (x, x)

        x_homo = self.lin_homo(x[0])
        x_hetero = self.lin_hetero(x[0])

        edge_index = edge_index.flip(0)
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x[1].size(0), x[0].size(0)))

        out_homo = self.propagate(edge_index, x=(x_homo, x[1]), size=size)
        out_hetero = self.propagate(edge_index, x=(x_hetero, x[1]), size=size)

        # 平均合并同配性和异配性
        out = (out_homo + out_hetero) / 2

        if self.root_weight and x[1] is not None:
            out += self.lin_r(x[1])

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out, out_homo, out_hetero

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)
