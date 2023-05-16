import torch.nn as nn

from layers.layers import get_dim_act
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
import manifolds


class Encoder(nn.Module):
    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        # return Embedding
        return output

class HGCF(Encoder):
    def __init__(self, c, args, adj_dim):
        super(HGCF, self).__init__(c)
        # add
        self.adj_dim = adj_dim
        # end add
        self.manifold = getattr(manifolds, "Hyperboloid")()
        assert args.num_layers > 1
        hgc_layers = []
        in_dim = out_dim = args.embedding_dim
        # add self.adj_dim
        hgc_layers.append(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, in_dim, out_dim, self.c, args.network, args.num_layers, self.adj_dim
            )
        )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(x, c=self.c)
        return super(HGCF, self).encode(x_hyp, adj)

class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        # add
        self.manifold = getattr(manifolds, "Hyperboloid")()
        # end add
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            # add
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat, c))
            # end add
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True