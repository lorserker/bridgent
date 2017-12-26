import sys
import numpy as np
import tensorflow as tf

from batcher import Batcher
from metrics import acc012

from tqdm import tqdm

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
display_step = 10000

learning_rate = 0.001

n_hidden_units = 128

bn_epsilon = 1e-5

l2_reg = float(sys.argv[2])

dropout_keep = float(sys.argv[1])

# set up neural network

X = tf.placeholder(tf.float32, shape=[None, n_h, n_w, n_c], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

conv1_w = tf.get_variable('c1w', shape=[1, 4, 4, 32], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
conv1_z = tf.nn.conv2d(X, filter=conv1_w, strides=[1,1,1,1], padding='SAME')
conv1_z_mean, conv1_z_var = tf.nn.moments(conv1_z, axes=[0,1,2], keep_dims=False)
conv1_scale = tf.Variable(tf.ones(conv1_z_mean.shape))
conv1_offset = tf.Variable(tf.zeros(conv1_z_mean.shape))
conv1_a = tf.nn.relu(tf.nn.batch_normalization(conv1_z, conv1_z_mean, conv1_z_var, conv1_offset, conv1_scale, bn_epsilon), name='conv1_a')

conv2_w = tf.get_variable('c2w', shape=[1, 4, 32, 64], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
conv2_z = tf.nn.conv2d(conv1_a, filter=conv2_w, strides=[1,1,1,1], padding='SAME')
conv2_z_mean, conv2_z_var = tf.nn.moments(conv2_z, axes=[0,1,2], keep_dims=False)
conv2_scale = tf.Variable(tf.ones(conv2_z_mean.shape))
conv2_offset = tf.Variable(tf.zeros(conv2_z_mean.shape))
conv2_a = tf.nn.relu(tf.nn.batch_normalization(conv2_z, conv2_z_mean, conv2_z_var, conv2_offset, conv2_scale, bn_epsilon), name='conv2_a')

conv3_w = tf.get_variable('c3w', shape=[1, 4, 64, 128], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
conv3_z = tf.nn.conv2d(conv2_a, filter=conv3_w, strides=[1,1,1,1], padding='SAME')
conv3_z_mean, conv3_z_var = tf.nn.moments(conv3_z, axes=[0,1,2], keep_dims=False)
conv3_scale = tf.Variable(tf.ones(conv3_z_mean.shape))
conv3_offset = tf.Variable(tf.zeros(conv3_z_mean.shape))
conv3_a = tf.nn.relu(tf.nn.batch_normalization(conv3_z, conv3_z_mean, conv3_z_var, conv3_offset, conv3_scale, bn_epsilon), name='conv3_a')

conv4_w = tf.get_variable('c4w', shape=[4, 4, 128, 512], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
conv4_z = tf.nn.conv2d(conv3_a, filter=conv4_w, strides=[1,1,1,1], padding='VALID')
conv4_z_mean, conv4_z_var = tf.nn.moments(conv4_z, axes=[0,1,2], keep_dims=False)
conv4_scale = tf.Variable(tf.ones(conv4_z_mean.shape))
conv4_offset = tf.Variable(tf.zeros(conv4_z_mean.shape))
conv4_a = tf.nn.relu(tf.nn.batch_normalization(conv4_z, conv4_z_mean, conv4_z_var, conv4_offset, conv4_scale, bn_epsilon), name='conv4_a')

conv_s_w = tf.get_variable('csw', shape=[1, 13, 4, 32], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
conv_s_z = tf.nn.conv2d(X, filter=conv_s_w, strides=[1,1,1,1], padding='VALID')
conv_s_z_mean, conv_s_z_var = tf.nn.moments(conv_s_z, axes=[0,1,2], keep_dims=False)
conv_s_scale = tf.Variable(tf.ones(conv_s_z_mean.shape))
conv_s_offset = tf.Variable(tf.zeros(conv_s_z_mean.shape))
conv_s_a = tf.nn.relu(tf.nn.batch_normalization(conv_s_z, conv_s_z_mean, conv_s_z_var, conv_s_offset, conv_s_scale, bn_epsilon), name='conv_s_a')

fc_in = tf.nn.dropout(
    tf.concat([tf.contrib.layers.flatten(conv4_a), tf.contrib.layers.flatten(conv_s_a)], axis=1, name='fc_in'), 
    keep_prob, 
    seed=seed
)
fc_w = tf.get_variable('fcw', shape=[fc_in.shape.as_list()[1], n_hidden_units], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
fc_z = tf.matmul(fc_in, fc_w)
fc_z_mean, fc_z_var = tf.nn.moments(fc_z, axes=[0], keep_dims=False)
fc_z_scale = tf.Variable(tf.ones(fc_z_mean.shape))
fc_z_offset = tf.Variable(tf.zeros(fc_z_mean.shape))
fc_a = tf.nn.relu(tf.nn.batch_normalization(fc_z, fc_z_mean, fc_z_var, fc_z_offset, fc_z_scale, bn_epsilon), name='fc_a')

fc_w_2 = tf.get_variable('fcw2', shape=[n_hidden_units, n_hidden_units], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
fc_z_2 = tf.matmul(fc_a, fc_w_2)
fc_z_2_mean, fc_z_2_var = tf.nn.moments(fc_z_2, axes=[0], keep_dims=False)
fc_z_2_scale = tf.Variable(tf.ones(fc_z_2_mean.shape))
fc_z_2_offset = tf.Variable(tf.zeros(fc_z_2_mean.shape))
fc_a_2 = tf.nn.relu(tf.nn.batch_normalization(fc_z_2, fc_z_2_mean, fc_z_2_var, fc_z_2_offset, fc_z_2_scale, bn_epsilon), name='fc_a_2')

fc_w_3 = tf.get_variable('fcw3', shape=[n_hidden_units, n_hidden_units], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
fc_z_3 = tf.matmul(fc_a_2, fc_w_3)
fc_z_3_mean, fc_z_3_var = tf.nn.moments(fc_z_3, axes=[0], keep_dims=False)
fc_z_3_scale = tf.Variable(tf.ones(fc_z_3_mean.shape))
fc_z_3_offset = tf.Variable(tf.zeros(fc_z_3_mean.shape))
fc_a_3 = tf.nn.relu(tf.nn.batch_normalization(fc_z_3, fc_z_3_mean, fc_z_3_var, fc_z_3_offset, fc_z_3_scale, bn_epsilon), name='fc_a_3')

fc_w_4 = tf.get_variable('fcw4', shape=[n_hidden_units, n_hidden_units], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
fc_z_4 = tf.matmul(fc_a_3, fc_w_4)
fc_z_4_mean, fc_z_4_var = tf.nn.moments(fc_z_4, axes=[0], keep_dims=False)
fc_z_4_scale = tf.Variable(tf.ones(fc_z_4_mean.shape))
fc_z_4_offset = tf.Variable(tf.zeros(fc_z_4_mean.shape))
fc_a_4 = tf.nn.relu(
    tf.add(
        tf.nn.batch_normalization(fc_z_4, fc_z_4_mean, fc_z_4_var, fc_z_4_offset, fc_z_4_scale, bn_epsilon),
        fc_a
    ),
    name='fc_a_4'
)

w_out = tf.get_variable('w_out', shape=[n_hidden_units, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
b_out = tf.Variable(np.zeros((1, 1)), dtype=tf.float32)
pred = tf.add(tf.matmul(fc_a_4, w_out), b_out, name='pred')

# define cost
weights = [conv1_w, conv2_w, conv3_w, conv4_w, conv_s_w, fc_w_2, fc_w_3, fc_w_4, w_out]
cost_reg = (1.0 / (2*batch_size)) * sum([tf.reduce_sum(tf.square(w)) for w in weights])
cost_pred = tf.reduce_mean(tf.squared_difference(pred, Y))
cost = cost_pred + l2_reg * cost_reg

# optimizer

train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(cost)

init = tf.global_variables_initializer()

batch = Batcher(n_examples, batch_size)
cost_train_batch = Batcher(n_examples, 10000)
cost_val_batch = Batcher(100000, 10000)

# run the session

model_path = sys.argv[3]

with tf.Session() as sess:
    sess.run(init)
    
    saver = tf.train.Saver(max_to_keep=100)

    for iteration in range(n_iterations // display_step):
        for i in range(display_step):
            x_batch, y_batch = batch.next_batch([X_train, y_train])
            train_step.run(feed_dict={X:x_batch, Y:y_batch, keep_prob:dropout_keep})
        
        saver.save(sess, model_path, global_step=iteration*display_step)

        sys.stdout.write('*')
        x_batch_c, y_batch_c = cost_train_batch.next_batch([X_train, y_train])
        x_batch_v, y_batch_v = cost_val_batch.next_batch([X_val, y_val])
        c = sess.run(cost, feed_dict={X: x_batch_c, Y: y_batch_c, keep_prob: 1.0})
        creg = sess.run(cost_reg, feed_dict={X: x_batch_c, Y: y_batch_c, keep_prob: 1.0})
        pred_train = sess.run(pred, feed_dict={X: x_batch_c, Y: y_batch_c, keep_prob: 1.0})
        pred_val = sess.run(pred, feed_dict={X: x_batch_v, Y: y_batch_v, keep_prob: 1.0})
        print('it={} cost={} creg={}'.format(iteration*display_step, c, l2_reg*creg))
        print(acc012(y_batch_c, pred_train))
        print(acc012(y_batch_v, pred_val))
        sys.stdout.flush()

    saver.save(sess, model_path, global_step=iteration*display_step)
