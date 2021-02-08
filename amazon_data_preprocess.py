import os
import gc
import numpy as np
import pandas as pd
from config import data_name, item_2_id
from sklearn.preprocessing import LabelEncoder

def exchange(x):
    if x != 0:
        x = str(x)
        x = x.replace('$', '')
        x = x.replace(',', '')
        return float(x)

def slice(x):
    if len(x) == 0:
        return 'unknown'
    return x[-1]

if __name__ == "__main__":
    item_id = item_2_id[data_name][0]

    print('reading data')
    ratings = pd.read_pickle('../raw_data/%s/ratings.pkl' % (data_name))
    item = pd.read_pickle('../raw_data/%s/%s.pkl' % (data_name, data_name))
    item.rename(columns={'asin': item_id}, inplace=True)

    print('labelencoding userid')
    lbe = LabelEncoder()
    ratings['userid'] = lbe.fit_transform(ratings['userid'])

    print('labelencoding itemid')
    lbe = LabelEncoder()
    unique_id = np.concatenate((ratings[item_id].unique(), item[item_id].unique()))
    lbe.fit(unique_id)
    ratings[item_id] = lbe.transform(ratings[item_id]) + 1
    item[item_id] = lbe.transform(item[item_id]) + 1

    pd.to_pickle(item, '../raw_data/%s/%s.pkl' % (data_name, data_name))
    pd.to_pickle(ratings, '../raw_data/%s/ratings.pkl' % (data_name))
