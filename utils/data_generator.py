import os
import pickle as pkl
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from utils.helper import sparse_mx_to_torch_sparse_tensor, normalize, default_device
from utils.log import set_color


class Data(object):
    def __init__(self, dataset, norm_adj, seed, test_ratio):
        pkl_path = os.path.join('./data/' + dataset)
        self.pkl_path = pkl_path
        self.dataset = dataset
        if dataset == 'ml-100k':
            self.user_item_list = self.load_pickle(
                os.path.join(pkl_path, 'user_item_list.pkl'))
            self.user_mapping = self.load_pickle(
                os.path.join(pkl_path, 'user_mapping.pkl'))
            self.item_mapping = self.load_pickle(
                os.path.join(pkl_path, 'item_mapping.pkl'))

            self.user_item_list, self.user_mapping, self.item_mapping = self.convert_to_inner_index(self.user_item_list,
                                                                                                    self.user_mapping,
                                                                                                    self.item_mapping)
            self.train_dict, self.test_dict = self.split_data_randomly(
                self.user_item_list, test_ratio, seed)
            self.num_users, self.num_items = len(
                self.user_mapping), len(self.item_mapping)

        elif dataset.split('-')[0] in ['Amazon', 'yelp']:
            self.user_item_list = self.load_pickle(
                os.path.join(pkl_path, 'user_item_list.pkl'))

            self.train_dict, self.test_dict = self.split_data_randomly(
                self.user_item_list, test_ratio, seed)


            self.num_users, self.num_items = len(self.user_item_list), max(
                [max(x) for x in self.user_item_list]) + 1

        # self.adj_train, self.features = self.generate_adj()
        self.adj_train = self.generate_adj()

        if eval(norm_adj):
            self.adj_train_norm = normalize(
                self.adj_train + sp.eye(self.adj_train.shape[0]))
            self.adj_train_norm = sparse_mx_to_torch_sparse_tensor(
                self.adj_train_norm)

        print(set_color("\nDataset Information:", 'pink'))
        print(set_color('Num_users', 'yellow') + set_color(' = ',
              'white') + set_color(str(self.num_users), 'blue'))
        print(set_color('Num_items', 'yellow') + set_color(' = ',
              'white') + set_color(str(self.num_items), 'blue'))
        print(set_color('Adjacency Matrix Shape', 'yellow') + set_color(' = ',
              'white') + set_color(str(self.adj_train.shape), 'blue'))


        tot_num_rating = sum([len(x) for x in self.user_item_list])
        del self.user_item_list

        print(set_color('Ratings', 'yellow') + set_color(' = ',
              'white') + set_color(str(tot_num_rating), 'blue'))
        print(set_color('Density', 'yellow') + set_color(' = ', 'white') +
              set_color(str(tot_num_rating / (self.num_users * self.num_items)), 'blue'))
        
        del tot_num_rating
        self.user_item_csr = self.generate_rating_matrix(
            [*self.train_dict.values()], self.num_users, self.num_items)
        self.user_item_csr_nonzero = self.user_item_csr.nonzero()
        del self.user_item_csr

    def generate_adj(self):
        user_item = np.zeros((self.num_users, self.num_items)).astype(int)
        for i, v in self.train_dict.items():
            user_item[i][v] = 1
        # features = sp.eye(user_item.shape[0])
        coo_user_item = sp.coo_matrix(user_item)

        del user_item

        print(set_color('\nGenerating Adjacency Matrix: ', 'green'))
        start = time.time()
        rows = np.concatenate(
            (coo_user_item.row, coo_user_item.transpose().row + self.num_users))
        cols = np.concatenate(
            (coo_user_item.col + self.num_users, coo_user_item.transpose().col))
        data = np.ones((coo_user_item.nnz * 2,))
        adj_csr = sp.coo_matrix(
            (data, (rows, cols))).tocsr().astype(np.float32)
        # # add
        # features = sp.eye(adj_csr.shape[0])
        # # edd add
        print(set_color("Time Elapsed: ", 'yellow') + set_color(str(time.time() - start), 'blue'))
        
        return adj_csr
        # return adj_csr, features

    def load_pickle(self, name):
        with open(name, 'rb') as f:
            return pkl.load(f, encoding='latin1')

    def split_data_randomly(self, user_records, test_ratio, seed):
        train_dict = {}
        test_dict = {}
        for user_id, item_list in enumerate(user_records):
            tmp_train_sample, tmp_test_sample = train_test_split(
                item_list, test_size=test_ratio, random_state=seed)

            train_sample = []
            for place in item_list:
                if place not in tmp_test_sample:
                    train_sample.append(place)

            test_sample = []
            for place in tmp_test_sample:
                test_sample.append(place)

            train_dict[user_id] = train_sample
            test_dict[user_id] = test_sample
        return train_dict, test_dict

    def convert_to_inner_index(self, user_records, user_mapping, item_mapping):
        inner_user_records = []
        user_inverse_mapping = self.generate_inverse_mapping(user_mapping)
        item_inverse_mapping = self.generate_inverse_mapping(item_mapping)

        for user_id in range(len(user_mapping)):
            real_user_id = user_mapping[user_id]
            item_list = list(user_records[real_user_id])
            for index, real_item_id in enumerate(item_list):
                item_list[index] = item_inverse_mapping[real_item_id]
            inner_user_records.append(item_list)

        return inner_user_records, user_inverse_mapping, item_inverse_mapping

    def generate_inverse_mapping(self, mapping):
        inverse_mapping = dict()
        for inner_id, true_id in enumerate(mapping):
            inverse_mapping[true_id] = inner_id
        return inverse_mapping

    def generate_rating_matrix(self, train_set, num_users, num_items):
        # three lists are used to construct sparse matrix
        row = []
        col = []
        data = []
        for user_id, article_list in enumerate(train_set):
            for article in article_list:
                row.append(user_id)
                col.append(article)
                data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        rating_matrix = csr_matrix(
            (data, (row, col)), shape=(num_users, num_items))

        return rating_matrix
