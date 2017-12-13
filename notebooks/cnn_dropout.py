import sys
import numpy as np
import tensorflow as tf

from batcher import Batcher
from metrics import acc012

seed = 1337

np.random.seed(seed)

X_train = np.load('../data/bin/train/deal.npy')
y_train = np.load('../data/bin/train/tricks_spades.npy')

X_val = np.load('../data/bin/val/deal.npy')
y_val = np.load('../data/bin/val/tricks_spades.npy')

n_examples = X_train.shape[0]

n_h = X_train.shape[1]
n_w = X_train.shape[2]
n_c = X_train.shape[3]

batch_size = 64
n_iterations = 1000000
display_step = 100

learning_rate = 0.001

n_hidden_units = 128

bn_epsilon = 1e-5

# set up neural network

X = tf.placeholder(tf.float32, shape=[None, n_h, n_w, n_c])
Y = tf.placeholder(tf.float32, shape=[None, 1])

keep_prob = tf.placeholder(tf.float32)

conv1_w = tf.get_variable('c1w', shape=[1, 4, 4, 32], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
conv1_z = tf.nn.conv2d(X, filter=conv1_w, strides=[1,1,1,1], padding='SAME')
conv1_z_mean, conv1_z_var = tf.nn.moments(conv1_z, axes=[0,1,2], keep_dims=False)
conv1_scale = tf.Variable(tf.ones(conv1_z_mean.shape))
conv1_offset = tf.Variable(tf.zeros(conv1_z_mean.shape))
conv1_a = tf.nn.relu(tf.nn.batch_normalization(conv1_z, conv1_z_mean, conv1_z_var, conv1_offset, conv1_scale, bn_epsilon))

conv2_w = tf.get_variable('c2w', shape=[1, 4, 32, 64], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
conv2_z = tf.nn.conv2d(conv1_a, filter=conv2_w, strides=[1,1,1,1], padding='SAME')
conv2_z_mean, conv2_z_var = tf.nn.moments(conv2_z, axes=[0,1,2], keep_dims=False)
conv2_scale = tf.Variable(tf.ones(conv2_z_mean.shape))
conv2_offset = tf.Variable(tf.zeros(conv2_z_mean.shape))
conv2_a = tf.nn.relu(tf.nn.batch_normalization(conv2_z, conv2_z_mean, conv2_z_var, conv2_offset, conv2_scale, bn_epsilon))

conv3_w = tf.get_variable('c3w', shape=[1, 4, 64, 128], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
conv3_z = tf.nn.conv2d(conv2_a, filter=conv3_w, strides=[1,1,1,1], padding='SAME')
conv3_z_mean, conv3_z_var = tf.nn.moments(conv3_z, axes=[0,1,2], keep_dims=False)
conv3_scale = tf.Variable(tf.ones(conv3_z_mean.shape))
conv3_offset = tf.Variable(tf.zeros(conv3_z_mean.shape))
conv3_a = tf.nn.relu(tf.nn.batch_normalization(conv3_z, conv3_z_mean, conv3_z_var, conv3_offset, conv3_scale, bn_epsilon))

conv4_w = tf.get_variable('c4w', shape=[4, 4, 128, 512], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
conv4_z = tf.nn.conv2d(conv3_a, filter=conv4_w, strides=[1,1,1,1], padding='VALID')
conv4_z_mean, conv4_z_var = tf.nn.moments(conv4_z, axes=[0,1,2], keep_dims=False)
conv4_scale = tf.Variable(tf.ones(conv4_z_mean.shape))
conv4_offset = tf.Variable(tf.zeros(conv4_z_mean.shape))
conv4_a = tf.nn.relu(tf.nn.batch_normalization(conv4_z, conv4_z_mean, conv4_z_var, conv4_offset, conv4_scale, bn_epsilon))

fc_in = tf.nn.dropout(tf.contrib.layers.flatten(conv4_a), keep_prob, seed=seed)
fc_w = tf.get_variable('fcw', shape=[fc_in.shape.as_list()[1], n_hidden_units], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
fc_z = tf.matmul(fc_in, fc_w)
fc_z_mean, fc_z_var = tf.nn.moments(fc_z, axes=[0], keep_dims=False)
fc_z_scale = tf.Variable(tf.ones(fc_z_mean.shape))
fc_z_offset = tf.Variable(tf.zeros(fc_z_mean.shape))
fc_a = tf.nn.dropout(
    tf.nn.relu(tf.nn.batch_normalization(fc_z, fc_z_mean, fc_z_var, fc_z_offset, fc_z_scale, bn_epsilon)),
    keep_prob,
    seed=seed)

w_out = tf.get_variable('w_out', shape=[n_hidden_units, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
b_out = tf.Variable(np.zeros((1, 1)), dtype=tf.float32)
pred = tf.add(tf.matmul(fc_a, w_out), b_out)

# define cost

cost_pred = tf.reduce_mean(tf.squared_difference(pred, Y))
cost = cost_pred

# optimizer

train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(cost)

init = tf.global_variables_initializer()

batch = Batcher(n_examples, batch_size)
cost_train_batch = Batcher(n_examples, 10000)
cost_val_batch = Batcher(100000, 10000)

# run the session

with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(n_iterations // display_step):
        for i in range(display_step):
            x_batch, y_batch = batch.next_batch([X_train, y_train])
            train_step.run(feed_dict={X:x_batch, Y:y_batch, keep_prob:0.7})
        
        sys.stdout.write('*')
        x_batch_c, y_batch_c = cost_train_batch.next_batch([X_train, y_train])
        x_batch_v, y_batch_v = cost_val_batch.next_batch([X_val, y_val])
        c = sess.run(cost, feed_dict={X: x_batch_c, Y: y_batch_c, keep_prob: 1.0})
        pred_train = sess.run(pred, feed_dict={X: x_batch_c, Y: y_batch_c, keep_prob: 1.0})
        pred_val = sess.run(pred, feed_dict={X: x_batch_v, Y: y_batch_v, keep_prob: 1.0})
        print('it={} cost={}'.format(iteration*display_step, c))
        print(acc012(y_batch_c, pred_train))
        print(acc012(y_batch_v, pred_val))
        sys.stdout.flush()
