import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def placeholder_inputs(batch_size, num_feature):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_feature))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, labels_pl

def get_model(feature, is_training, bn_decay=None):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  # MLP on global point cloud vector
  layers = {}
  feature = tf.squeeze(feature)
  layer_name = 'ft_'
  
  net = tf_util.fully_connected(feature, 512, bn=True, is_training=is_training,
                                scope=layer_name + 'fc2', bn_decay=bn_decay,
                                activation_fn = tf.nn.relu)
  layers[layer_name + 'fc2'] = net
  
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                         scope=layer_name + 'dp2')
  
  net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                scope=layer_name + 'fc3', bn_decay=bn_decay,
                                activation_fn = tf.nn.relu)
  layers[layer_name + 'fc3'] = net
  
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                        scope=layer_name + 'dp3')
  net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc4')
  layers[layer_name + 'fc4'] = net

  return net, layers


def get_loss(pred, label):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=40)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
      print (res1.shape)
      print (res1)

      print (res2.shape)
      print (res2)