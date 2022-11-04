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
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = pointclouds.get_shape()[0].value
    num_point = pointclouds.get_shape()[1].value
    end_points = {}
    
    l0_xyz = pointclouds
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True, reuse=reuse)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2', reuse=reuse)
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3', reuse=reuse)

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util_verif.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, reuse=reuse)
    net = tf_util_verif.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1', reuse=reuse)
    net = tf_util_verif.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay, reuse=reuse)
    net = tf_util_verif.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2', reuse=reuse)

    # net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')                            # original
    # net = tf_util_verif.fully_connected(net, num_class, activation_fn=None, scope='fc3', reuse=reuse)  # Bernardo
    net = tf_util_verif.fully_connected(net, 256, activation_fn=None, scope='fc3', reuse=reuse)          # Bernardo

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
def contrastive_loss(left_feature, right_feature, label, margin=1.0):
    """
    Compute the contrastive loss as in

    L = 0.5 * Y * D^2 + 0.5 * (Y-1) * {max(0, margin - D)}^2

    **Parameters**
     left_feature: First element of the pair
     right_feature: Second element of the pair
     label: Label of the pair (0 or 1)
     margin: Contrastive margin

    **Returns**
     Return the loss operation

    """

    label = tf.to_float(label)
    one = tf.constant(1.0)

    dist = euclidean_distance(left_feature, right_feature)
    first_part = tf.multiply(one-label, dist)# (Y-1)*(d)

    max_part = tf.square(tf.maximum(margin-dist, 0))
    second_part = tf.multiply(label, max_part)  # (Y) * max(margin - d, 0)

    # loss = 0.5 * tf.reduce_mean(first_part + second_part)
    loss = first_part + second_part
    return loss

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
def get_loss(pred1, pred2, label, end_points1, end_points2):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    individual_losses = contrastive_loss(pred1, pred2, label, margin=1.0)
    
    classify_loss = tf.reduce_mean(individual_losses)
    # classify_loss = individual_loss

    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    tf.add_to_collection('individual_losses', individual_losses)
    return classify_loss, individual_losses


if __name__=='__main__':
    
    with tf.Graph().as_default():
        pointclouds_pl1, pointclouds_pl2, labels_pl = placeholder_inputs(batch_size=32, num_point=1024)

        is_training_pl = tf.placeholder(tf.bool, shape=())

        pred1, end_points1, pred2, end_points2 = get_model(pointclouds_pl1, pointclouds_pl2, is_training_pl, bn_decay=None)    # Bernardo
        pred, individual_loss = get_loss(pred1, pred2, labels_pl, end_points1, end_points2)

        data1 = np.ones((32,1024,3),dtype=np.float32)
        data2 = np.random.rand(32,1024,3)
        label = np.zeros((32,),dtype=np.int32)

        is_training = False
        sess = tf.Session()
        with sess.as_default():
            print_op = tf.Print(pointclouds_pl1, [pred])
            with tf.control_dependencies([print_op]):
                out = tf.add(pred, pred)
            sess.run(tf.global_variables_initializer())

            pred, individual_loss, pred1, pred2 = sess.run([pred, individual_loss, pred1, pred2],
                                                     feed_dict={pointclouds_pl1: data2,
                                                     pointclouds_pl2: data2,
                                                     labels_pl: label,
                                                     is_training_pl:True})

            print('total loss:', pred)
            print('individual_loss:', individual_loss)
            print('pred1:', pred1)
            print('pred2:', pred2)
            print('pred2.shape:', pred2.shape)

