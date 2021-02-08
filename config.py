data_name = 'game'

train_batch_size = 1024
test_batch_size = 2048

n_users = {'mv1M':6040, 'mv25M':7045, 'toys':208180, 'book':52643, 'game':55222, 'music':16565, 'phone':27878, 'CD':75257}
n_items = {'mv1M':3952, 'mv25M':209163, 'toys':78772, 'book':91599, 'game':71976, 'music':77625, 'phone':284012, 'CD':410164}
item_2_id = {'mv1M':['movieid'], 'mv25M':['movieid'], 'toys':['toy_id'], 'book':['bookid'], 'game':['game_id'], 'music':['music_id'], 'phone':['item_id'], 'CD':['item_id']}
