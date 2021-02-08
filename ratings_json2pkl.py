import pandas as pd
import argparse
from config import data_name, item_2_id

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='game')
args = parser.parse_args()

item_id = item_2_id[data_name]
ratings = pd.read_json('../raw_data/%s/Video_Games_5.json' % (args.dataset), lines=True)
rating = ratings[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
rating.rename(columns={'reviewerID':'userid', 'asin':item_id, 'overall':'rating', 'unixReviewTime':'timestamp'}, inplace=True)
rating.to_pickle('../raw_data/%s/raw_ratings.pkl' % (args.dataset))
