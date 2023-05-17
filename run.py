
import time
import traceback
from datetime import datetime

import numpy as np

from config import parser
from eval_metrics import recall_at_k_gpu
from models.base_models import HGCFModel
# from models.base_models import HGATModel
from rgd.rsgd import RiemannianSGD
from utils import get_tensorboard, early_stop
from utils.data_generator import Data
from utils.helper import default_device, set_seed, sparse_mx_to_torch_sparse_tensor
from utils.log import Logger, set_color
from utils.sampler import WarpSampler
import itertools
import heapq
import torch
import scipy.sparse as sp


def train(model):
    print(set_color("\nTraining: ", 'green'))
    optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, momentum=args.momentum)

    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(set_color("Total number of parameters", 'yellow') +
        set_color(' = ', 'white'), set_color(str(tot_params), 'blue'))

    num_pairs = data.adj_train.count_nonzero() // 2
    num_batches = int(num_pairs / args.batch_size) + 1
    print(set_color("Number of Batches", 'yellow') + set_color(' = ',
        'white'), set_color(str(num_batches), 'blue') + '\n')

    tb_writer = get_tensorboard()

    step_count = 0
    best_score = [[np.NINF, np.NINF, np.NINF, np.NINF],
                [np.NINF, np.NINF, np.NINF, np.NINF]]

    # add
    # 考虑是否需要修改
    # data.adj_train = sparse_mx_to_torch_sparse_tensor(data.adj_train + sp.eye(data.adj_train.shape[0]))
    # data.adj_train = sparse_mx_to_torch_sparse_tensor(data.adj_train)
    # end add

    for epoch in range(1, args.epochs + 1):
        avg_loss = 0.
        t = time.time()
        for batch in range(num_batches):
            triples = sampler.next_batch()
            model.train()
            optimizer.zero_grad()
            embeddings = model.encode(data.adj_train_norm)
            train_loss = model.compute_loss(embeddings, triples)
            train_loss.backward()
            optimizer.step()
            avg_loss += train_loss / num_batches
            # print("Batch: {}".format(batch))
        

        tb_writer.add_scalar("Loss/Train", avg_loss, epoch)

        if args.log:
            log.write('Train:{:3d}\t{:.2f}\n'.format(epoch, avg_loss))
        else:
            print(set_color('Epoch:', 'pink') + '\t' +
                set_color('{:04d}'.format(epoch), 'blue') + '\t' +
                set_color('Loss:', 'pink') + '\t' +
                set_color('{:.3f}'.format(avg_loss), 'blue') + '\t' +
                set_color('Time:', 'pink') + '\t' +
                set_color('{}s'.format(time.time() - t), 'blue'))

        stop_flag = 0

        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            start = time.time()
            embeddings = model.encode(data.adj_train_norm)
            pred_matrix_gpu = model.predict(embeddings, data)
            print("Encode:\t{}s\t".format(time.time() - start) +
                "Pred:\t{}s".format(time.time() - start))
            results = eval_rec(pred_matrix_gpu, data)

            if args.log:
                log.write('Test:{:3d}\t{:.3f}\t{:.4f}\t{:.3f}\t{:.4f}\n'.format(epoch + 1, results[0][1], results[0][2],
                                                                                results[-1][1], results[-1][2]))
            else:

                print(set_color("R_GPU@:", 'green') + '\t' +
                    '\t'.join([str(round(x, 4)) for x in results[0]]))
                print(set_color("N_GPU@:", 'green') + '\t' +
                    '\t'.join([str(round(x, 4)) for x in results[-1]]))

            tb_writer.add_scalar('Valid_score', results[0][1], epoch)
            best_score, step_count, stop_flag = early_stop(
                results, best_score, step_count)

        if stop_flag:
            print(set_color("Best Result:", 'pink'))
            print("R_GPU@:\t" + '\t'.join([str(round(x, 4))
                                        for x in best_score[0]]))
            print("N_GPU@:\t" + '\t'.join([str(round(x, 4))
                                        for x in best_score[-1]]))
            break

    sampler.close()


def argmax_top_k(a, top_k=50):
    topk_score_items = []
    for i in range(len(a)):
        topk_score_item = heapq.nlargest(top_k, zip(a[i], itertools.count()))
        topk_score_items.append([x[1] for x in topk_score_item])
    return topk_score_items


def ndcg_func_cpu(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        # rank = rank.cpu().numpy()
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum(
            [1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)


def ndcg_func_gpu(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        rank = rank.cpu().numpy()
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum(
            [1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)


def eval_rec(pred_matrix_gpu, data):
    topk = 50
    pred_matrix_gpu[data.user_item_csr_nonzero] = np.NINF
    ind_gpu = torch.topk(pred_matrix_gpu, k=topk, dim=1)[1]
    arr_ind_gpu = pred_matrix_gpu[np.arange(
        len(pred_matrix_gpu))[:, None], ind_gpu]
    arr_ind_gpu_argsort = torch.argsort(arr_ind_gpu, dim=1, descending=True)
    del arr_ind_gpu
    pred_list_gpu = ind_gpu[np.arange(len(pred_matrix_gpu))[
        :, None], arr_ind_gpu_argsort]
    del ind_gpu
    del arr_ind_gpu_argsort
    recall_gpu = []
    for k in [5, 10, 20, 50]:
        recall_gpu.append(recall_at_k_gpu(data.test_dict, pred_list_gpu, k))

    all_ndcg_gpu = ndcg_func_gpu([*data.test_dict.values()], pred_list_gpu)
    ndcg_gpu = [all_ndcg_gpu[x-1] for x in [5, 10, 20, 50]]
    return recall_gpu, ndcg_gpu


if __name__ == '__main__':
# ----------------------------------------------------------------
# config init
    args = parser.parse_args()
    if args.log:
        now = datetime.now()
        now = now.strftime('%m-%d_%H-%M-%S')
        log = Logger(args.log, now)
        for arg in vars(args):
            log.write(arg + '=' + str(getattr(args, arg)) + '\n')
    set_seed(args.seed)
# ----------------------------------------------------------------
# generate data
    data = Data(args.dataset, args.norm_adj, args.seed, args.test_ratio)
    total_edges = data.adj_train.count_nonzero()
    args.n_nodes = data.num_users + data.num_items
    args.feat_dim = data.adj_train_norm.shape[1]
# ----------------------------------------------------------------
# sampler init
    sampler = WarpSampler((data.num_users, data.num_items),data.adj_train, args.batch_size, args.num_neg)
# ----------------------------------------------------------------
# model init
    model = HGCFModel((data.num_users, data.num_items), args)
    # model = HGATModel((data.num_users, data.num_items), args)
    model = model.to(default_device())
    print(str(model))
    print(set_color('\nModel is running on: ', 'green') +
        set_color(str(next(model.parameters()).device), 'red'))
# ----------------------------------------------------------------
# train
    try:
        train(model)
    except Exception:
        sampler.close()
        traceback.print_exc()
