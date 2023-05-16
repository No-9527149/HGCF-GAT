import numpy as np
import torch
import torch.nn as nn

import manifolds
import models.encoders as encoders
from utils.helper import default_device
import scipy.sparse as sp


# add


class HGATModel(nn.Module):

    def __init__(self, users_items, args):
        super(HGATModel, self).__init__()

        self.c = torch.tensor([args.c]).to(default_device())
        self.manifold = getattr(manifolds, "Hyperboloid")()
        self.nnodes = args.n_nodes
        # add
        args.feat_dim = args.feat_dim + 1
        # end add
        self.encoder = getattr(encoders, "GAT")(self.c, args)

        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.num_layers = args.num_layers

        self.args = args

    def encode(self, adj_train, adj_train_norm):
        o = torch.zeros((adj_train.shape[0], 1)).to(default_device())
        x = adj_train.to_dense().to(default_device())
        x = torch.cat((o, x), dim=1)
        if torch.cuda.is_available():
            adj_train_norm = adj_train_norm.to(default_device())
            x = x.to(default_device())
        h = self.encoder.encode(x, adj_train_norm)
        return h

    def decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        return sqdist

    def compute_loss(self, embeddings, triples):
        train_edges = triples[:, [0, 1]]

        sampled_false_edges_list = [triples[:, [0, 2 + i]]
                                    for i in range(self.args.num_neg)]

        pos_scores = self.decode(embeddings, train_edges)

        neg_scores_list = [self.decode(embeddings, sampled_false_edges) for sampled_false_edges in
                            sampled_false_edges_list]
        neg_scores = torch.cat(neg_scores_list, dim=1)

        loss = pos_scores - neg_scores + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss

    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix_gpu = torch.zeros((num_users, num_items)).type(
            torch.float32).to(default_device())
        for i in range(num_users):
            emb_in = h[i, :]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)
            emb_out = h[np.arange(num_users, num_users + num_items), :]
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            probs = sqdist.detach() * -1
            probs_matrix_gpu[i] = torch.reshape(probs, [-1, ])
        return probs_matrix_gpu
