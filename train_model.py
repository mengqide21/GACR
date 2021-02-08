import argparse
import os
import sys
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
from time import time
from data_loader import Data
from config import train_batch_size, test_batch_size, n_users, n_items
from test_tool import test_model
from model import GACR

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='game')
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--hist_len', type=int, default=5)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--emb_dim', type=int, default=16)
parser.add_argument('--layer_size', default='[16,16,16,16]')
parser.add_argument('--hid', default='[32,16]')
parser.add_argument('--l2_reg', type=float, default=1e-5)
parser.add_argument('--logi_reg', type=float, default=0.4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', default='[0.1,0.1,0.1,0.1]')
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--frac', type=float, default=0.9)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
tfconfig = tf.ConfigProto()
#config.gpu_options.allow_growth = True
tfconfig.gpu_options.per_process_gpu_memory_fraction = args.frac
sess = tf.Session(config=tfconfig)

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def early_stopping(cur_value, best_value, stopping_step, order='acc', max_step=100):
    # for auc or logloss (acc and dec)
    if (order == 'acc' and cur_value >= best_value) or (order == 'dec' and cur_value <= best_value):
        stopping_step = 0
        best_value = cur_value
    else:
        stopping_step += 1

    if stopping_step >= max_step:
        print("early stopping at: {} log:{}".format(max_step, cur_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

if __name__ == '__main__':
    data_name = args.dataset
    data_generator = Data(data_name, n_users[data_name], n_items[data_name])
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    num_layer = len(eval(args.layer_size))

    # generate Laplacian matrix
    norm_adj = data_generator.get_adj_mat()
    config['norm_adj'] = norm_adj
    
    hidden = eval(args.hid)

    model = GACR(data_config=config, layer_size=args.layer_size, history_length=args.hist_len, lr=args.lr, emb_dim=args.emb_dim, hidden1_dim=hidden[0],
                        hidden2_dim=hidden[1], l2_weight=args.l2_reg, logical_weight=args.logi_reg, interact_type='concat', sim_scale=10)

    sess.run(tf.global_variables_initializer())
    cur_best_auc = 0.
    print('initialization')

    # training
    loss_loger, auc_loger, log_loger = [], [], []
    stopping_step = 0
    should_stop = False

    users_train, histories_train, targets_train, labels_train = data_generator.load_data(train_flag='train') 
    users_test, histories_test, targets_test, labels_test = data_generator.load_data(train_flag='test')

    for i in range(args.epoch):
        t0 = time()
        loss = 0.
        n_batch = data_generator.n_train // train_batch_size + 1

        batch_test = data_generator.batch_iterator(users_test, histories_test, targets_test, labels_test, history_len=args.hist_len, batch_size=test_batch_size)
        batch_iter = data_generator.batch_iterator(users_train, histories_train, targets_train, labels_train, history_len=args.hist_len, batch_size=train_batch_size)

        for idx, batch in enumerate(batch_iter):
            user_batch, hist_batch, target_batch, label_batch = batch
            _, batch_loss = sess.run([model.opt, model.loss], 
                                        feed_dict={model.users: user_batch, model.pos_items: target_batch, model.hist_items: hist_batch, 
                                        model.labels: label_batch, model.mess_dropout: eval(args.dropout)})
            loss += batch_loss

        if np.isnan(loss) == True:
            print('loss is nan.')
            sys.exit()

        # test
        t1 = time()
        ret = test_model(sess, model, batch_test, num_layer, drop_flag=True)

        loss_loger.append(loss)
        auc_loger.append(ret['auc'])
        log_loger.append(ret['logloss'])

        performance = 'epoch %d [%.1fs]: train loss==[%.4f], auc=[%.4f], logloss=[%.4f]' % \
                        (i, t1-t0, loss, ret['auc'], ret['logloss'])
        print(performance)

        cur_best_auc, stopping_step, should_stop = early_stopping(ret['auc'], cur_best_auc,
                                                                    stopping_step, order='acc', max_step=5)

        if should_stop == True:
            break
    
    # best result
    aucs = np.array(auc_loger)
    loglosses = np.array(log_loger)

    best_auc_0 = max(aucs)
    idx = list(aucs).index(best_auc_0)

    best_performance = "best performance is auc=%.4f, logloss=%.4f" % (best_auc_0, loglosses[idx])
    print(best_performance)
