"""
The network model of linked dynamic graph CNN for segmentation. We borrow the edge 
convolutional operation from the DGCNN and design our own network architecture.
Reference code: https://github.com/WangYueFt/dgcnn
@author: Kuangen Zhang

"""
import tensorflow as tf
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../'))
import tf_util

def calc_ldgcnn_feature(point_cloud, is_training, weight_decay, bn_decay = None):
    # B: batch size; N: number of points, C: channels; k: number of nearest neighbors
    # point_cloud: B*N*3
    layer_name = 'ldgcnn_seg'
    k = 30
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
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope=layer_name + 'dgcnn1', bn_decay=bn_decay, is_dist=True)
    # net: B*N*k*64
    net = tf_util.conv2d(net, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope=layer_name + 'dgcnn1_2', bn_decay=bn_decay, is_dist=True)
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
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope=layer_name + 'dgcnn2', bn_decay=bn_decay, is_dist=True)
    # net: B*N*k*64
    net = tf_util.conv2d(net, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope=layer_name + 'dgcnn2_2', bn_decay=bn_decay, is_dist=True)
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
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope=layer_name + 'dgcnn3', bn_decay=bn_decay, is_dist=True)
    # net: B*N*k*64
    net = tf_util.conv2d(net, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope=layer_name + 'dgcnn3_2', bn_decay=bn_decay, is_dist=True)
    
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net
    
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    
    # net: B*N*1*195
    net = tf.concat([point_cloud, net1, net2, net3], axis=-1) # extend operation
    # edge_feature: B*N*k*390
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
    
    # net: B*N*k*128
    net = tf_util.conv2d(edge_feature, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope=layer_name + 'dgcnn4', bn_decay=bn_decay, is_dist=True)
    # net: B*N*k*128
    net = tf_util.conv2d(net, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope=layer_name + 'dgcnn4_2', bn_decay=bn_decay, is_dist=True)
    # net_concat: B*N*1*128
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net4 = net
    
    # net_concat: B*N*1*323
    net_concat = tf.concat([point_cloud, net1, net2, net3, net4], axis=-1)
    # net: B*N*1*1024
    net = tf_util.conv2d(net_concat, 1024, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope=layer_name + 'agg', bn_decay=bn_decay, is_dist=True)
    return net, net_concat


def get_model(point_cloud, input_label, is_training, cat_num, part_num, \
    batch_size, num_point, weight_decay, bn_decay=None):

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
  
    # Extract linked features and global features simultaneously.
    # net: B*N*1*1024; net_concat: B*N*1*323;
    net, net_concat = calc_ldgcnn_feature(point_cloud, is_training,weight_decay, bn_decay)
    out_max = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')
    
    # Because this is part segmentation experiment, we can input the category 
    # of the whole object and estimate the part of each point. 
    one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
    one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 64, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='one_hot_label_expand', bn_decay=bn_decay, is_dist=True)
    # out_max: B*1*1*1088;
    out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
    # expand: B*N*1*1088;
    expand = tf.tile(out_max, [1, num_point, 1, 1])

    # expand: B*N*1*1411;
    concat = tf.concat(axis=3, values=[expand, net_concat])
  
    layers = {}
    layers['concat'] = concat
  
  
    # Zero padding can increase the performance of my network,
    # but I do not knot the reason
    num_channel = concat.get_shape()[-1].value
    concat = tf.concat([concat, tf.zeros([batch_size,num_point, 1, 2*num_channel])], axis = -1)
    
    # Fully connected layers: classifier
    # net2: B*N*512
    net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay, is_dist=True)
    # Each element is kept or dropped independently, and the drop rate is (1-0.6).
    net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
    
    # net2: B*N*256
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
    net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
    
    # net2: B*N*128
    net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
    # net2: B*N*part_num
    net2 = tf_util.conv2d(net2, part_num, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
            bn=False, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)

    net2 = tf.reshape(net2, [batch_size, num_point, part_num])
    return net2, layers


def get_loss(seg_pred, seg):
  per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
  seg_loss = tf.reduce_mean(per_instance_seg_loss)
  per_instance_seg_pred_res = tf.argmax(seg_pred, 2)
  
  return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res
