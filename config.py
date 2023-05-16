import argparse

from hgcn_utils.train_utils import add_flags_from_config

# Amazon-CD
config_args = {
    'Training Config:': {
        # 如果加log，应该加到哪里
        'log': (None, 'None for no logging'),
        # 学习率，怎么调整
        'lr': (0.001, 'learning rate'),
        # batch-size是否过大？
        'batch-size': (10000, 'batch size'),
        'epochs': (500, 'maximum number of epochs to train for'),
        # L2正则化的长度？为什么L2正则化需要一个长度？
        'weight-decay': (0.005, 'l2 regularization strength'),
        # 优化器中的momentum是什么作用？
        'momentum': (0.95, 'momentum in optimizer'),
        'seed': (1234, 'seed for data split and training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        # 如果像RecBole一样，每个epoch都做eval？
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        # add
        'dropout': (0.0, 'dropout probability'),
        # end add
    },
    'Model Config:': {
        'embedding_dim': (50, 'user item embedding dimension'),
        # 什么scale？
        'scale': (0.1, 'scale for init'),
        # 为什么有embedding_dim，还有一个dim？
        'dim': (50, 'embedding dimension'),
        'network': ('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'),
        'c': (1, 'hyperbolic radius, set to None for trainable curvature'),
        # 编码器中的隐藏层数量
        'num-layers': (4,  'number of hidden layers in encoder'),
        # 度量学习损失中的保证金值？
        'margin': (0.1, 'margin value in the metric learning loss'),
        # add
        'act': ('None', 'which activation function to use (or None for no activation)'),
        'task': ('rec', 'which tasks to train on, can be any of [lp, nc, rec]'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        # end add
    },
    'Data Config: ': {
        'dataset': ('Amazon-CD', 'which dataset to use'),
        # 只有一个负样本？其实已经有负样本采样了？
        'num_neg': (1, 'number of negative samples'),
        # 用于链路预测的测试边的比例
        'test_ratio': (0.2, 'proportion of test edges for link prediction'),
        # 是否对邻居矩阵进行【行正则化】
        'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
    }
}

# Amazon-Book
# config_args = {
#     'training_config': {
#         'log': (None, 'None for no logging'),
#         'lr': (0.001, 'learning rate'),
#         'batch-size': (10000, 'batch size'),
#         'epochs': (500, 'maximum number of epochs to train for'),
#         'weight-decay': (0.0005, 'l2 regularization strength'),
#         'momentum': (0.95, 'momentum in optimizer'),
#         'seed': (1234, 'seed for data split and training'),
#         'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
#         'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
#     },
#     'model_config': {
#         'embedding_dim': (50, 'user item embedding dimension'),
#         'scale': (0.1, 'scale for init'),
#         'dim': (50, 'embedding dimension'),
#         'network': ('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'),
#         'c': (1, 'hyperbolic radius, set to None for trainable curvature'),
#         'num-layers': (4,  'number of hidden layers in encoder'),
#         'margin': (0.1, 'margin value in the metric learning loss'),
#     },
#     'data_config': {
#         'dataset': ('Amazon-Book', 'which dataset to use'),
#         'num_neg': (1, 'number of negative samples'),
#         'test_ratio': (0.2, 'proportion of test edges for link prediction'),
#         'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
#     }
# }

# # Yelp
# config_args = {
#     'training_config': {
#         'log': (None, 'None for no logging'),
#         'lr': (0.001, 'learning rate'),
#         # 'batch-size': (4096, 'batch size'),
#         'batch-size': (10000, 'batch size'),
#         'epochs': (500, 'maximum number of epochs to train for'),
#         'weight-decay': (0.001, 'l2 regularization strength'),
#         'momentum': (0.95, 'momentum in optimizer'),
#         'seed': (2023, 'seed for data split and training'),
#         'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
#         'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
#     },
#     'model_config': {
#         'embedding_dim': (50, 'user item embedding dimension'),
#         'scale': (0.1, 'scale for init'),
#         'dim': (50, 'embedding dimension'),
#         'network': ('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'),
#         'c': (1, 'hyperbolic radius, set to None for trainable curvature'),
#         'num-layers': (4,  'number of hidden layers in encoder'),
#         'margin': (0.2, 'margin value in the metric learning loss'),
#     },
#     'data_config': {
#         'dataset': ('yelp', 'which dataset to use'),
#         'num_neg': (1, 'number of negative samples'),
#         'test_ratio': (0.2, 'proportion of test edges for link prediction'),
#         'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
#     }
# }

# init
# config_args = {
#     'training_config': {
#         'log': (True, 'None for no logging'),
#         'lr': (0.001, 'learning rate'),
#         'batch-size': (10000, 'batch size'),
#         # 'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
#         'epochs': (500, 'maximum number of epochs to train for'),
#         'weight-decay': (0.001, 'l2 regularization strength'),
#         'momentum': (0.95, 'momentum in optimizer'),
#         'seed': (1234, 'seed for data split and training'),
#         'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
#         'eval-freq': (20, 'how often to compute val metrics (in epochs)'),
#     },
#     'model_config': {
#         'embedding_dim': (50, 'user item embedding dimension'),
#         'scale': (0.1, 'scale for init'),
#         'dim': (50, 'embedding dimension'),
#         'network': ('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'),
#         'c': (1, 'hyperbolic radius, set to None for trainable curvature'),
#         'num-layers': (4,  'number of hidden layers in encoder'),
#         'margin': (0.1, 'margin value in the metric learning loss'),
#     },
#     'data_config': {
#         'dataset': ('Amazon-CD', 'which dataset to use'),
#         'num_neg': (1, 'number of negative samples'),
#         'test_ratio': (0.2, 'proportion of test edges for link prediction'),
#         'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
#     }
# }

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    # config_dict：分别为trainning、model、data config字典
    # _：分别为字符串trainning/model/data_config
    parser = add_flags_from_config(_, parser, config_dict)
