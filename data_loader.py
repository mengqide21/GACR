import numpy as np
import random as rd
import scipy.sparse as sp
import tensorflow as tf
import pandas as pd
import os
from config import item_2_id

class Data(object):
    def __init__(self, data_name, n_users, n_items):
        self.data_name = data_name
        self.path = '../sampled_data/%s' % (self.data_name)
        self.item_id = item_2_id[data_name][0]

        train_file = self.path + '/user_hist_train.pkl'
        test_file = self.path + '/user_hist_test.pkl'

        self.n_users, self.n_items = n_users, n_items
        self.n_train, self.n_test = 0, n_users
        self.neg_pools = {}

        self.exist_users = []
        self.R = sp.dok_matrix((self.n_users+1, self.n_items+1), dtype=np.float32)

        self.train_items = pd.read_pickle(train_file)
        for uid, items in self.train_items.items():
            self.exist_users.append(uid)
            self.n_train += len(items)
            for i in items:
                self.R[uid, i] = 1
        
        self.test_set = pd.read_pickle(test_file)

    def get_adj_mat(self):
        if os.path.exists(self.path + '/adj_mat.npz'):
            norm_adj_mat = sp.load_npz(self.path + '/adj_mat.npz')

        else:
            norm_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/adj_mat.npz', norm_adj_mat)
        return norm_adj_mat

    def create_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items + 2, self.n_users + self.n_items + 2), dtype=np.float32)
        # sparse matrix
        adj_mat = adj_mat.tolil() 
        R = self.R.tolil()

        adj_mat[:self.n_users+1, self.n_users+1:] = R
        adj_mat[self.n_users+1:, :self.n_users+1] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj(mat):
            rowsum = np.array(mat.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(mat)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj(adj_mat + sp.eye(adj_mat.shape[0]))

        return norm_adj_mat.tocsr()
    
    # generate sample for CTR task
    def batch_iterator(self, users, hist_items, target_items, labels, history_len=5, batch_size=1024, shuffle=True):
        # padding for history
        hist_data = tf.keras.preprocessing.sequence.pad_sequences(hist_items, maxlen=history_len,
                                                            value=0)
        user_data = np.array(users)
        target_data = np.array(target_items)
        label_data = np.array(labels)

        # shuffle data
        if shuffle:
            indices = np.random.permutation(range(len(labels)))
            hist_data = hist_data[indices]
            user_data = user_data[indices]
            target_data = target_data[indices]
            label_data = label_data[indices]

        num_batch = int((len(labels) - 1) / batch_size + 1)
        for i in range(num_batch):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(labels))
            yield user_data[start: end], hist_data[start: end], target_data[start: end], label_data[start: end]
    
    def load_data(self, train_flag='train'):
        print('loading %s data' % (train_flag))
        users, histories, targets, labels = [], [], [], []

        raw_samples = pd.read_pickle('../sampled_data/' + self.data_name + '/raw_sample_3_' + train_flag + '.pkl')
        hist_dic = pd.read_pickle('../sampled_data/' + self.data_name + '/user_hist_' + train_flag + '.pkl')
        for _, i in raw_samples.iterrows():
            users.append(i['userid'])
            targets.append(i[self.item_id])
            labels.append(i['label'])
            hist_ = hist_dic[i['userid']]
            histories.append(hist_)
        
        return users, histories, targets, labels