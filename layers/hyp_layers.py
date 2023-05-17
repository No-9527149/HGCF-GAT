import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
from utils.helper import default_device


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, network, num_layers, adj_dim):
        super(HyperbolicGraphConvolution, self).__init__()
        self.agg = HypAgg(manifold, c_in, out_features, network, num_layers, adj_dim)

    def forward(self, input):
        x, adj = input
        h = self.agg.forward(x, adj)
        output = h, adj
        return output


class StackGCNs(Module):

    def __init__(self, num_layers, adj_dim):
        super(StackGCNs, self).__init__()

        self.num_gcn_layers = num_layers - 1

        self.a = nn.Parameter(torch.zeros(adj_dim, adj_dim)).to(default_device())
        nn.init.xavier_normal_(self.a.data, gain = 0.5)

    def plainGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        # a = nn.Parameter(torch.zeros(size=(1, 2 * )))
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return output[-1]

    def resSumGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        # a = 1 * 41342
        for i in range(self.num_gcn_layers):
            temp = torch.spmm(adj, output[i]).to(default_device())
            temp = self.a.mm(temp)
            temp = temp.squeeze()
            temp = torch.sigmoid(temp)
            temp = temp / 5 - 0.1
            output.append(temp)
            # output.append((torch.sigmoid(a.mm(torch.spmm(adj, output[i])).squeeze())) / 5 - 0.1)
        # output = 41342 * 50
        return sum(output[1:])

    def resAddGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        if self.num_gcn_layers == 1:
            return torch.spmm(adj, x_tangent)
        for i in range(self.num_gcn_layers):
            if i == 0:
                output.append(torch.spmm(adj, output[i]))
            else:
                output.append(output[i] + torch.spmm(adj, output[i]))
        return output[-1]

    def denseGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            if i > 0:
                output.append(sum(output[1:i + 1]) + torch.spmm(adj, output[i]))
            else:
                output.append(torch.spmm(adj, output[i]))
        return output[-1]


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, network, num_layers, adj_dim):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.adj_dim = adj_dim
        self.stackGCNs = getattr(StackGCNs(num_layers, self.adj_dim), network)

    def forward(self, x, adj):
        # 投影到切线空间
        x_tangent = self.manifold.logmap0(x, c=self.c)
        # 在切线空间中做stackGCNs
        # 返回一个tensor
        output = self.stackGCNs((x_tangent, adj))
        # tensor先到H，再proj
        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


