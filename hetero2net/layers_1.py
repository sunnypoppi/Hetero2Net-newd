from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm, to_torch_csr_tensor
from torch import nn
import torch
import matplotlib.pyplot as plt

def to_sparse_tensor(edge_index, size):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=size
    )
# 继承消息传播机制， 用于实现异构图的GNN传播机制
# 利用相关性，解耦为“同配性”和“异配性”两部分
class DisenConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project
        self.bn = nn.BatchNorm1d(1)


        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # 同配性信息
        self.lin_homo = Linear(in_channels[0], out_channels, bias=bias)
        # 异配性信息
        self.lin_hetero = Linear(in_channels[0], out_channels, bias=bias)
        # 根特征
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        # **Gating Network**: 让模型学习同配性和异配性的重要程度
        self.gate = Linear(in_channels[0], 1)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_homo.reset_parameters()
        self.lin_hetero.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
        self.gate.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # 计算两种特征
        x_homo = self.lin_homo(x[0])
        x_hetero = self.lin_hetero(x[0])

        # 转换成稀疏矩阵，进行邻居信息传播
        edge_index = to_sparse_tensor(edge_index.flip(0), size=(x[1].size(0), x[0].size(0)))
        out_homo = self.propagate(edge_index, x=(x_homo, x[1]), size=size)
        out_hetero = self.propagate(edge_index, x=(x_hetero, x[1]), size=size)

        # **动态权重 α**
        alpha = torch.sigmoid(self.bn(self.gate(x[1])))  # 计算时用 x[1]，保证维度匹配
        #alpha = torch.sigmoid(self.gate(torch.cat([x[0], x[1]], dim=-1)))
        alpha = alpha.squeeze(-1).unsqueeze(-1).expand(-1, self.out_channels)


        # 让 α 决定如何加权同配性和异配性信息
        out = alpha * out_homo + (1 - alpha) * out_hetero
        # alpha 互换位置 91.40 91.55

        if self.root_weight and x[1] is not None:
            out = out + self.lin_r(x[1])

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out, out_homo, out_hetero

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
