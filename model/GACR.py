import tensorflow as tf
import numpy as np

class GACR(object):
    def __init__(self, data_config, layer_size, history_length=5,
                        lr=0.01, emb_dim=8, hidden1_dim=16, hidden2_dim=8, batch_size=1024, l2_weight=1e-6, logical_weight=0.1, 
                        interact_type='sum', sim_scale=10, verbose=1):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100
        self.norm_adj = data_config['norm_adj']

        self.lr = lr
        self.logical_weight = logical_weight

        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.hist_len = history_length
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        self.weight_size = eval(layer_size)
        self.n_layers = len(self.weight_size)

        self.l2 = l2_weight
        self.interact_type = interact_type
        self.activation = tf.nn.relu
        self.sim_scale = sim_scale

        self.verbose = verbose

        # placeholder
        self.users = tf.placeholder(tf.int32, shape=(None,), name='user_input')
        self.pos_items = tf.placeholder(tf.int32, shape=(None,), name='target_input')
        self.hist_items = tf.placeholder(tf.int32, shape=(None, self.hist_len), name='history_input')
        self.labels = tf.placeholder(tf.int32, shape=(None,), name='label_input')

        # dropout
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None], name='mess_dropout')

        # initialization of parameters
        self.weights = self.init_weights()
        self.user_emb, self.item_embedding = self.enhance_embedding()

        self.u_embeddings = tf.nn.embedding_lookup(self.user_emb, self.users)
        self.u_g_embeddings = tf.expand_dims(self.u_embeddings, 1)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_embedding, self.pos_items)
        self.pos_i_g_embeddings = tf.expand_dims(self.pos_i_embeddings, 1)
        self.hist_i_g_embeddings = tf.nn.embedding_lookup(self.item_embedding, self.hist_items)

        # event interaction
        self.encoder = interaction_module(self.u_g_embeddings, self.hist_i_g_embeddings, self.hidden1_dim,
                                        self.hidden2_dim, activation=self.activation)
        self.encoder_pos = interaction_module(self.u_g_embeddings, self.pos_i_g_embeddings,
                                            self.hidden1_dim, self.hidden2_dim, activation=self.activation)
        
        #logic reason
        self.not_encoder = not_module(self.encoder, self.hidden1_dim, self.hidden2_dim, activation=self.activation)
        self.or_cell = or_module(self.hidden1_dim, self.hidden2_dim)
        self.or_encoder, _ = tf.nn.dynamic_rnn(self.or_cell, self.not_encoder[:, 1:, :], initial_state=self.not_encoder[:, 0, :],
                                            dtype=tf.float32)
        self.or_encoder_last = self.or_encoder[:, -1, :]

        self.or_encoder_pos, _ = tf.nn.dynamic_rnn(self.or_cell, self.encoder_pos, initial_state=self.or_encoder_last, dtype=tf.float32)

        # prediction
        self.similarity_pos = cos_sim(self.or_encoder_pos, self.True_emb)
        self.similarity_pos = tf.squeeze(self.similarity_pos, 1)
        self.probability = tf.nn.sigmoid(self.similarity_pos * self.sim_scale)

        # loss
        self.mf_loss, self.emb_loss = self.create_loss(self.u_g_embeddings, self.pos_i_g_embeddings, self.labels)
        self.target_loss = self.mf_loss + self.emb_loss

        all_events = [self.encoder, self.encoder_pos, self.not_encoder, self.or_encoder, self.or_encoder_pos]
        all_events = tf.concat(all_events, axis=1)
        self.logical_loss = self.logical_regularizer(all_events)

        global_step = tf.train.get_or_create_global_step()
        self.loss = self.target_loss + self.logical_weight * self.logical_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=global_step)

    def init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()
        self.True_emb = tf.get_variable(name='truth_vector', shape=[1, 1, self.hidden2_dim], initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32, trainable=False)
        all_weights['user_embedding'] = tf.get_variable(name='user_embedding', shape=[self.n_users+1, self.emb_dim], initializer=tf.contrib.layers.xavier_initializer())
        all_weights['item_embedding'] = tf.get_variable(name='item_embedding', shape=[self.n_items+1, self.emb_dim], initializer=tf.contrib.layers.xavier_initializer())

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_self_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_self_%d' % k)
            all_weights['b_self_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_self_%d' % k)

            all_weights['W_inter_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_inter_%d' % k)
            all_weights['b_inter_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_inter_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights

    def _split_A_hat(self, mat):
        res = []
        fold_len = (self.n_users + self.n_items + 2) // self.n_fold

        for i in range(self.n_fold):
            start = i * fold_len
            if i == self.n_fold -1:
                end = self.n_users + self.n_items + 2
            else:
                end = (i + 1) * fold_len
            res.append(self._convert_sp_mat_to_sp_tensor(mat[start:end]))

        return res
    
    def enhance_embedding(self):
        L_mat = self._split_A_hat(self.norm_adj)

        cur_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        final_embeddings = [cur_embeddings]

        for k in range(0, self.n_layers):
            temp_emb = []
            for f in range(self.n_fold):
                temp_emb.append(tf.sparse_tensor_dense_matmul(L_mat[f], cur_embeddings))
            ui_embeddings = tf.concat(temp_emb, 0)

            # conduct message passing and integration in matrix
            a = tf.nn.leaky_relu(
                tf.matmul(ui_embeddings, self.weights['W_self_%d' % k]) + self.weights['b_self_%d' % k])
            b = tf.nn.leaky_relu(
                tf.matmul(tf.multiply(cur_embeddings, ui_embeddings), self.weights['W_inter_%d' % k]) + self.weights['b_inter_%d' % k])
            cur_embeddings = a + b

            # dropout.
            cur_embeddings = tf.nn.dropout(cur_embeddings, 1 - self.mess_dropout[k])

            # normalize
            norm_embeddings = tf.nn.l2_normalize(cur_embeddings, axis=1)

            # add current layer
            final_embeddings += [norm_embeddings]

        final_embeddings = tf.concat(final_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(final_embeddings, [self.n_users+1, self.n_items+1], 0)
        return u_g_embeddings, i_g_embeddings

    def create_loss(self, users, target_items, labels):
        scores = self.probability
        loss = tf.reduce_mean(tf.losses.log_loss(labels=labels, predictions=scores))

        trainable_variables = tf.trainable_variables()
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in trainable_variables])

        l2_loss = l2_loss/self.batch_size
        emb_loss = self.l2 * l2_loss

        return loss, emb_loss
    
    def logical_regularizer(self, all_events):
        False_emb = not_module(self.True_emb, self.hidden1_dim, self.hidden2_dim, activation=self.activation)
        not_event = not_module(all_events, self.hidden1_dim, self.hidden2_dim,
                                activation=self.activation)
        double_not_event = not_module(not_event, self.hidden1_dim, self.hidden2_dim,
                                        activation=self.activation)
        l1 = tf.reduce_mean(1 + cos_sim(not_event, all_events))
        l2 = tf.reduce_mean(1 - cos_sim(double_not_event, all_events))

        event_or_F = self.or_cell(all_events, False_emb)
        l3 = tf.reduce_mean(1 - cos_sim(event_or_F, all_events))

        event_or_T = self.or_cell(all_events, self.True_emb)
        l4 = tf.reduce_mean(1 - cos_sim(event_or_T, self.True_emb))

        event_or_event = self.or_cell(all_events, all_events)
        l5 = tf.reduce_mean(1 - cos_sim(event_or_event, event_or_event))

        event_or_not_event = self.or_cell(all_events, not_event)
        l6 = tf.reduce_mean(1 - cos_sim(event_or_not_event, self.True_emb))

        return l1 + l2 + l3 + l4 + l5 + l6

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)


def interaction_module(user, item, hidden1, hidden2, activation=tf.nn.relu):
    users = tf.tile(user, [1, tf.shape(item)[1], 1])
    merge_vec = tf.concat([users, item], axis=-1)

    event = tf.layers.dense(merge_vec, hidden1, activation=activation, name='interaction_1', reuse=tf.AUTO_REUSE)
    event = tf.layers.dense(event, hidden2, name='interaction_2', reuse=tf.AUTO_REUSE)
    return event


def not_module(input, hidden1, hidden2, activation=tf.nn.relu):
    not_encoder = tf.layers.dense(input, hidden1, activation=activation, name='not_1', reuse=tf.AUTO_REUSE)
    not_encoder = tf.layers.dense(not_encoder, hidden2, name='not_2', reuse=tf.AUTO_REUSE)
    return not_encoder


def cos_sim(a, b):
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(a), axis=-1))
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(b), axis=-1))
    res = tf.reduce_sum(tf.multiply(a, b), axis=-1) / (a_norm * b_norm)
    return res

class or_module(tf.nn.rnn_cell.RNNCell):
    def __init__(self, hidden1, hidden2, activation=None, reuse=tf.AUTO_REUSE, name=None):
        super(or_module, self).__init__(_reuse=reuse, name=name)
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.activation = tf.nn.relu

    @property
    def state_size(self):
        return self.hidden2

    @property
    def output_size(self):
        return self.hidden2

    def build(self, inputs_shape):
        self.layer_1 = tf.layers.Dense(self.hidden1, activation=self.activation, name="or_1")
        self.layer_2 = tf.layers.Dense(self.hidden2, name="or_2")
        self.built = True

    def call(self, inputs, hid_state):
        #hidden = tf.concat([inputs, hid_state], axis=-1)
        hidden = inputs + hid_state
        hidden = self.layer_1(hidden)
        res = self.layer_2(hidden)
        return res, res