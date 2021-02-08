import numpy as np
import multiprocessing
import heapq
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import test_batch_size
from sklearn.metrics import log_loss, roc_auc_score

def test_model(sess, model, batch_test, num_layer, drop_flag=False, batch_test_flag=False):
    result = {'logloss': 0, 'auc': 0}
    n_batches = 0

    for _, batch in enumerate(batch_test):
        user_batch, hist_batch, target_batch, label_batch = batch

        if drop_flag == False:
            pred_batch = sess.run(model.probability, {model.users: user_batch,
                                                            model.pos_items: target_batch,
                                                            model.hist_items: hist_batch})
        else:
            pred_batch = sess.run(model.probability, {model.users: user_batch,
                                                            model.pos_items: target_batch,
                                                            model.hist_items: hist_batch,
                                                            model.mess_dropout: [0.] * num_layer})

        #user_batch_rating_uid = zip(pred_batch, user_batch, label_batch)
        logloss = log_loss(label_batch, pred_batch)
        auc = roc_auc_score(label_batch, pred_batch)
        n_batches += 1

        result['auc'] += auc
        result['logloss'] += logloss

    result['auc'] /= n_batches
    result['logloss'] /= n_batches

    return result