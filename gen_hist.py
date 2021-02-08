import pandas as pd
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
import argparse
import os
from tqdm import tqdm
from config import data_name, item_2_id

itemid = item_2_id[data_name][0]

def gen_hist(uid, t):
    t.sort_values('timestamp', inplace=True, ascending=True)
    hist = []
    for _, row in t.iterrows():
        item_id = row[itemid]
        hist.append(item_id)

    return uid, hist[:-2], hist[:-1]

def applyParallel(df_grouped, func, n_jobs, backend='multiprocessing'):
    """Use Parallel and delayed """  
    results = Parallel(n_jobs=n_jobs, verbose=4, backend=backend)(
        delayed(func)(name, group) for name, group in df_grouped)

    return {k: v for k, v, _ in results}, {k:v for k, _, v in results}

if __name__ == '__main__':
    if not os.path.exists('../sampled_data'):
        os.mkdir('../sampled_data')
    if not os.path.exists('../sampled_data/' + str(data_name)):
        os.mkdir('../sampled_data/' + str(data_name))

    print('reading data')
    data = pd.read_pickle('../raw_data/%s/ratings.pkl' % (data_name))

    # generate each user's history
    data_grouped = data.groupby('userid')
    user_hist_train, user_hist_test = applyParallel(data_grouped, gen_hist, n_jobs=5)

    pd.to_pickle(user_hist_train, '../sampled_data/' + data_name + '/user_hist_train.pkl')
    pd.to_pickle(user_hist_test, '../sampled_data/' + data_name + '/user_hist_test.pkl')

    print('gen hist done')