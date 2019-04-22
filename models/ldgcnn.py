"""
The network model of linked dynamic graph CNN. We borrow the edge 
convolutional operation from the DGCNN and design our own network architecture.
Reference code: https://github.com/WangYueFt/dgcnn
@author: Kuangen Zhang

"""
import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
import pointfly as pf
#%%
# Add input placeholder
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl 

# Input point cloud and output the global feature
def calc_ldgcnn_feature(point_cloud, is_training, bn_decay = None):
    # B: batch size; N: number of points, C: channels; k: number of nearest neighbors
    # point_cloud: B*N*3
    k = 20
    
    # adj_matrix: B*N*N
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    # Find the indices of knearest neighbors.
    nn_idx = tf_util.knn(adj_matrix, k=k)
    
    point_cloud = tf.expand_dims(point_cloud, axis = -2)
    
    # Edge_feature: B*N*k*6
    # The vector in the last dimension represents: (Xc,Yc,Zc, Xck - Xc, Yck-Yc, Yck-zc)
    # (Xc,Yc,Zc) is the central point. (Xck - Xc, Yck-Yc, Yck-zc) is the edge vector.
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
    
    # net: B*N*k*64
    # The kernel size of CNN is 1*1, and thus this is a MLP with sharing parameters.
    net = tf_util.conv2d(edge_feature, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)
    
    # net: B*N*1*64
    # Extract the biggest feature from k convolutional edge features.     
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net
    
    # adj_matrix: B*N*N
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    
    # net: B*N*1*67 
    # Link the Hierarchical features.
    net = tf.concat([point_cloud, net1], axis=-1)
    
    # edge_feature: B*N*k*134
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)
    
    # net: B*N*k*64
    net = tf_util.conv2d(edge_feature, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)
    # net: B*N*1*64
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net
    
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    
    # net: B*N*1*131
    net = tf.concat([point_cloud, net1, net2], axis=-1)
    
    # edge_feature: B*N*k*262
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
    
    # net: B*N*k*64
    net = tf_util.conv2d(edge_feature, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)
    # net: B*N*1*64
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net
    
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    
    # net: B*N*1*195
    net = tf.concat([point_cloud, net1, net2, net3], axis=-1)
    # edge_feature: B*N*k*390
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
    
    # net: B*N*k*128
    net = tf_util.conv2d(edge_feature, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='dgcnn4', bn_decay=bn_decay)
    # net: B*N*1*128
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net4 = net
    
    # input: B*N*1*323
    # net: B*N*1*1024
    net = tf_util.conv2d(tf.concat([point_cloud, net1, net2, net3, 
                                    net4], axis=-1), 1024, [1, 1], 
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='agg', bn_decay=bn_decay)
    # net: B*1*1*1024
    net = tf.reduce_max(net, axis=1, keep_dims=True)
    # net: B*1024
    net = tf.squeeze(net)
    return net

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    layers = {}
    
    # Extract global feature
    net = calc_ldgcnn_feature(point_cloud, is_training, bn_decay)
    
    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1]) 
    layers['global_feature'] = net
    
    # Fully connected layers: classifier
    # net: B*512
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    layers['fc1'] = net
    # Each element is kept or dropped independently, and the drop rate is 0.5.
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                           scope='dp1')
    
    # net: B*256
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    layers['fc2'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp2')
    
    # net: B*40
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
    layers['fc3'] = net
    return net,layers


def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # Change the label from an integer to the one_hot vector.
    labels = tf.one_hot(indices=label, depth=40)
    # Calculate the loss based on cross entropy method.
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    # Calculate the mean loss of a batch input.
    classify_loss = tf.reduce_mean(loss)
    return classify_loss

if __name__=='__main__':
    # Test the network architecture
    batch_size = 2
    num_pt = 1024
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