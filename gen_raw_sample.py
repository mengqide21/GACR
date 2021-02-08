import pandas as pd
import os
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
from config import data_name, item_2_id
from sklearn.preprocessing import LabelEncoder

def gen_sample(grouped, state='train'):
    group = grouped.sort_values('timestamp', ascending=True)
    if state == 'train':
        group = group.iloc[-2:-1]
    else:
        group = group.iloc[-1:]
    return group

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

parser = argparse.ArgumentParser()
parser.add_argument('--state', type=str, default='test')
args = parser.parse_args()

item_id = item_2_id[data_name][0]

print('reading')
ratings = pd.read_pickle('../raw_data/%s/ratings.pkl' % (data_name))

print('n_users: ', ratings['userid'].max())
print('n_items: ', ratings[item_id].max())
input()

print('grouping')
grouped = ratings.groupby('userid')

results = Parallel(n_jobs=5,backend='multiprocessing', verbose=4)(delayed(gen_sample)(gp, args.state) for _, gp in grouped)
raw_sample = pd.concat(results, axis=0)

raw_sample.loc[raw_sample['rating']<4, 'rating'] = 0
raw_sample.loc[raw_sample['rating']!=0, 'rating'] = 1
raw_sample.rename(columns={'rating':'label'}, inplace=True)
ensureDir('../sampled_data/' + data_name + '/raw_sample_3_' + args.state + '.pkl')
pd.to_pickle(raw_sample, '../sampled_data/' + data_name + '/raw_sample_3_' + args.state + '.pkl')