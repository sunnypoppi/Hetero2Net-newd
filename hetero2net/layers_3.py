import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Linear
from torch_sparse import SparseTensor
from torch_geometric.utils import spmm, to_torch_csr_tensor

class DisenConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="mean", normalize=False, root_weight=True, bias=True):
        super().__init__(aggr=aggr, node_dim=0)
        self._user_args = ['x', 'x_raw']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_homo = Linear(in_channels[0], out_channels, bias=bias)  # 同配路径
        self.lin_hetero = Linear(in_channels[0], out_channels, bias=bias)  # 异配路径
        self.attr_gate = nn.Linear(out_channels, 1)  # 属性差异调制分支

        if root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)  # 节点自身的权重

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_homo.reset_parameters()
        self.lin_hetero.reset_parameters()
        self.attr_gate.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: tuple, edge_index: torch.Tensor, size: tuple = None):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        print(f"x[0] shape: {x[0].shape}, x[1] shape: {x[1].shape}")

        x_homo = self.lin_homo(x[0])
        x_hetero = self.lin_hetero(x[0])

        print(f"x_homo shape: {x_homo.shape}, x_hetero shape: {x_hetero.shape}")

        # Set size to be the number of nodes in x[0] and x[1]
        if size is None:
            size = (x[0].size(0), x[0].size(0))  # Assuming x[0] and x[1] correspond to the same number of nodes

        print(f"Size: {size}")

        # Check if the size of x[1] has changed
        if x[1].size(0) != size[1]:
            size = (x[0].size(0), x[1].size(0))

        out_homo = self.propagate(edge_index, x=(x_homo, x[1]), x_raw=(x[0], x[1]), size=size)
        out_hetero = self.propagate(edge_index, x=(x_hetero, x[1]), x_raw=(x[0], x[1]), size=size)

        out = out_homo + out_hetero

        if self.root_weight and x[1] is not None:
            out += self.lin_r(x[1])

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out, out_homo, out_hetero


    def message(self, x_j, x_i, x_raw_j, x_raw_i):
        # 属性差异调制：基于输入特征（未线性变换）
        attr_diff = torch.abs(x_raw_j - x_raw_i).mean(dim=1, keepdim=True)
        # 调整属性差异的维度，使其与 attr_gate 兼容
        #print("x_raw_j",x_raw_j.size(),"x_raw_i",x_raw_i.size())
        #print("x_j",x_j.size(),"x_i",x_i.size())
        #print(f"attr_diff shape before view: {attr_diff.shape}")
        #attr_diff = attr_diff.view(-1, self.out_channels)
        
        # 计算 diff_score，并确保与 x_j 的维度匹配
        diff_score = torch.sigmoid(attr_diff)  # [E, 1902]
        #diff_score = 1 - diff_score

        # 对 diff_score 进行处理，以确保它与 x_j 维度匹配
        if diff_score.size(1) != x_j.size(1):
            diff_score = diff_score.mean(dim=1, keepdim=True)  # 将 diff_score 压缩为 [E, 1]

        print("属性差异门控值:", diff_score.size())
        print("diff_score",diff_score)
        return  x_j * diff_score    # 将差异映射为信息门控

    # def aggregate(self, inputs, index, dim_size=None):
    #     # 确保inputs和index的维度匹配
    #     if inputs.dim() != index.dim():
    #         index = index.view(-1, 1).expand_as(inputs)  # 将index的维度调整为与inputs匹配
        
    #     # 使用scatter_add_进行聚合操作
    #     out = torch.zeros_like(inputs).scatter_add_(dim=0, index=index, src=inputs)
    #     return out
    # def propagate(self, edge_index, size=None, **kwargs):
    #     return super().propagate(edge_index=edge_index, size=size, **kwargs)

    # def message_and_aggregate(self, adj_t: SparseTensor, x: tuple) -> torch.Tensor:
    #     adj_t = adj_t.set_value(None, layout=None)
    #     return spmm(adj_t, x[0], reduce=self.aggr)
