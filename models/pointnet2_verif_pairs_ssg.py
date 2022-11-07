"""
    PointNet++ Model for point clouds verification (BERNARDO)
"""

from __future__ import print_function

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util_verif
from pointnet_util_verif import pointnet_sa_module



# Bernardo
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl1 = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_pl2 = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl1, pointclouds_pl2, labels_pl


# Bernardo
def get_backbone(pointclouds, is_training, bn_decay=None, reuse=False):
    with tf.name_scope("model"):
        batch_size = pointclouds.get_shape()[0].value
        num_point = pointclouds.get_shape()[1].value
        end_points = {}
        
        l0_xyz = pointclouds
        l0_points = None
        end_points['l0_xyz'] = l0_xyz

        with tf.variable_scope("pointnet_sa_module1") as scope:
            # Set abstraction layers
            # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
            # So we only use NCHW for layer 1 until this issue can be resolved.
            l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, use_nchw=True, scope=scope, reuse=reuse)
        
        with tf.variable_scope("pointnet_sa_module2") as scope:
            l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope=scope, reuse=reuse)
        
        with tf.variable_scope("pointnet_sa_module3") as scope:
            l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope=scope, reuse=reuse)

        # Fully connected layers
        net = tf.reshape(l3_points, [batch_size, -1])

        with tf.variable_scope("fully_connected1") as scope:
            net = tf_util_verif.fully_connected(net, 512, bn=True, is_training=is_training, scope=scope, bn_decay=bn_decay, reuse=reuse)
        
        with tf.variable_scope("dropout1") as scope:
            net = tf_util_verif.dropout(net, keep_prob=0.5, is_training=is_training, scope=scope, reuse=reuse)
        
        with tf.variable_scope("fully_connected2") as scope:
            net = tf_util_verif.fully_connected(net, 256, bn=True, is_training=is_training, scope=scope, bn_decay=bn_decay, reuse=reuse)
        
        with tf.variable_scope("dropout2") as scope:
            net = tf_util_verif.dropout(net, keep_prob=0.5, is_training=is_training, scope=scope, reuse=reuse)

        with tf.variable_scope("fully_connected3") as scope:
            # net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')                            # original
            # net = tf_util_verif.fully_connected(net, num_class, activation_fn=None, scope='fc3', reuse=reuse)  # Bernardo
            net = tf_util_verif.fully_connected(net, 256, activation_fn=None, scope=scope, reuse=reuse)          # Bernardo

    return net, end_points


# Bernardo
# def get_model(point_cloud, is_training, bn_decay=None, embedding_size=256):
def get_model(pointclouds_pl1, pointclouds_pl2, is_training, bn_decay=None):
    net1, end_points1 = get_backbone(pointclouds_pl1, is_training, bn_decay=None, reuse=False)
    net2, end_points2 = get_backbone(pointclouds_pl2, is_training, bn_decay=None, reuse=True)

    return net1, end_points1, net2, end_points2


# Bernardo
def euclidean_distance(x, y):
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1)
    # return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))
    return tf.sqrt(sum_square)

# Bernardo
# cosine similarity = (a . b) / ||a|| ||b||
def cosine_distance(x, y):
    cos_sim = tf.reduce_sum(tf.multiply(x, y), axis=1) / ( tf.multiply(tf.norm(x, axis=1), tf.norm(y, axis=1)) )
    cos_dist = 1 - cos_sim
    return cos_dist


def classify_pairs(dist, thresh):
    condition = tf.less_equal(dist, tf.constant(thresh))
    # condition = tf.less_equal(dist, tf.reduce_mean(dist))    # just for tests
    
    pred_labels = tf.where(condition, tf.ones_like(dist), tf.zeros_like(dist))
    return pred_labels


'''
    Compute the contrastive loss as in
    L = Y * D^2   +   (1-Y) * {max(0, margin - D)}^2

    **Parameters**
     left_feature: First element of the pair
     right_feature: Second element of the pair
     label: Label of the pair (0 or 1)
     margin: Contrastive margin

    **Returns**
     Return the loss operation
'''
# Bernardo
def contrastive_loss(left_feature, right_feature, true_label, margin):
    true_label = tf.to_float(true_label)

    # distances = euclidean_distance(left_feature, right_feature)
    distances = cosine_distance(left_feature, right_feature)
    pred_labels = classify_pairs(distances, thresh=margin)    

    # Bernardo
    first_part = true_label * tf.square(distances)
    one = tf.ones_like(true_label)
    zero = tf.zeros_like(true_label)
    second_part = (one - true_label) * tf.square(tf.maximum(margin - distances, zero))

    loss = first_part + second_part
    return loss, distances, pred_labels

''' # original reference
def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss
'''

# Bernardo
def get_loss(pred1, pred2, true_label, end_points1, end_points2):
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    individual_losses, distances, pred_labels = contrastive_loss(pred1, pred2, true_label, margin=0.5)
    
    # classify_loss = tf.reduce_mean(individual_losses)
    classify_loss = tf.reduce_sum(individual_losses)
    # classify_loss = individual_loss

    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    tf.add_to_collection('individual_losses', individual_losses)
    tf.add_to_collection('distances', distances)
    tf.add_to_collection('pred_labels', pred_labels)
    return classify_loss, individual_losses, distances, pred_labels


if __name__=='__main__':
    
    with tf.Graph().as_default():
        pointclouds_pl1, pointclouds_pl2, labels_pl = placeholder_inputs(batch_size=32, num_point=1024)

        is_training_pl = tf.placeholder(tf.bool, shape=())

        pred1, end_points1, pred2, end_points2 = get_model(pointclouds_pl1, pointclouds_pl2, is_training_pl, bn_decay=None)    # Bernardo
        total_loss, individual_loss, distances, pred_labels = get_loss(pred1, pred2, labels_pl, end_points1, end_points2)

        # data1 = np.ones((32,1024,3),dtype=np.float32)
        data1 = np.random.rand(32,1024,3)
        data2 = np.random.rand(32,1024,3)
        # data2 = data1
        true_labels = np.zeros((32,),dtype=np.int32)
        # true_labels = np.ones((32,),dtype=np.int32)

        is_training = False
        sess = tf.Session()
        with sess.as_default():
            print_op = tf.Print(pointclouds_pl1, [total_loss])
            with tf.control_dependencies([print_op]):
                out = tf.add(total_loss, total_loss)
            sess.run(tf.global_variables_initializer())

            total_loss, individual_loss, distances, pred_labels, pred1, pred2 = sess.run([total_loss, individual_loss, distances, pred_labels, pred1, pred2],
                                                            feed_dict={pointclouds_pl1: data1,
                                                                       pointclouds_pl2: data2,
                                                                       labels_pl: true_labels,
                                                                       is_training_pl: is_training})

            print('total loss:', total_loss)
            print('individual_loss:', individual_loss)
            print('distances:', distances)
            print('true_labels:', true_labels)
            print('pred_labels:', pred_labels)
            print('pred_labels.shape:', pred_labels.shape)
            print('true_labels == pred_labels:', true_labels == pred_labels)
            print('np.sum(true_labels == pred_labels)):', np.sum(true_labels == pred_labels))
            # print('pred1:', pred1)
            # print('pred2:', pred2)
            # print('pred2.shape:', pred2.shape)

