import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
from .extutils import create_ext_edges


class XGAT(MessagePassing):
    r"""

    Extended attention convolutional graph.

    """

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, extProb=1, bias=True, **kwargs):
        super(XGAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.extProb = extProb

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))

        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.att_ext = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.att_ext)

        zeros(self.bias)

    def forward(self, x, edge_index, xref, size=None):

        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),

                 None if x[1] is None else torch.matmul(x[1], self.weight))

        all_x, all_edges, anchor2, size2 = create_ext_edges(x, edge_index, xref)

        all_x = self.propagate(all_edges, size=size, x=all_x, anchor2=anchor2, size2=size2)

        return all_x[:-size2]

    def message(self, edge_index_i, x_i, x_j, size_i, anchor2, size2):

        # Compute attention coefficients for normal edges.
        ox_i = x_i[:anchor2]
        ox_j = x_j[:anchor2]
        oedege_index_i = edge_index_i[:anchor2]
        ox_j = ox_j.view(-1, self.heads, self.out_channels)
        if ox_i is None:
            oalpha = (ox_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            ox_i = ox_i.view(-1, self.heads, self.out_channels)
            oalpha = (torch.cat([ox_i, ox_j], dim=-1) * self.att).sum(dim=-1)

        oalpha = F.leaky_relu(oalpha, self.negative_slope)
        oalpha = softmax(oalpha, oedege_index_i, size_i)

        # Sample attention coefficients stochastically.
        oalpha = F.dropout(oalpha, p=self.dropout, training=self.training)

        # Compute attention coefficients for ext edges.
        ex_i = x_i[anchor2:]
        ex_j = x_j[anchor2:]

        eedege_index_i = edge_index_i[anchor2:]

        ex_j = ex_j.view(-1, self.heads, self.out_channels)

        if ex_i is None:
            ealpha = (ex_j * self.att_ext[:, :, self.out_channels:]).sum(dim=-1)
        else:
            ex_i = ex_i.view(-1, self.heads, self.out_channels)
            ealpha = (torch.cat([ex_i, ex_j], dim=-1) * self.att_ext).sum(dim=-1)

        ealpha = F.leaky_relu(ealpha, self.negative_slope)

        # ealpha = softmax(ealpha, eedege_index_i, size_i)
        # ealpha = ealpha * 0.01

        # Sample attention coefficients stochastically.
        ealpha = F.dropout(ealpha, p=self.dropout, training=self.training)

        alpha = torch.cat((oalpha, ealpha), dim=0)
        # alpha = alpha.view(-1, self.heads, 1)

        return x_j * alpha

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
