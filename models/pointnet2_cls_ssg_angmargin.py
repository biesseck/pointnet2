"""
    PointNet++ Model for point clouds classification
"""

from __future__ import print_function

import os
import sys
import math
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util_angmargin
from pointnet_util import pointnet_sa_module

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None, num_class=0):
    with tf.name_scope("model"):
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        end_points = {}
        l0_xyz = point_cloud
        l0_points = None
        end_points['l0_xyz'] = l0_xyz

        # Set abstraction layers
        # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
        # So we only use NCHW for layer 1 until this issue can be resolved.
        with tf.variable_scope("pointnet_sa_module1") as scope:
            l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
        
        with tf.variable_scope("pointnet_sa_module2") as scope:
            l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        
        with tf.variable_scope("pointnet_sa_module3") as scope:
            l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # Fully connected layers
        net = tf.reshape(l3_points, [batch_size, -1])

        with tf.variable_scope("fully_connected1") as scope:
            net, weights_fc1 = tf_util_angmargin.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
            
        with tf.variable_scope("dropout1") as scope:
            net = tf_util_angmargin.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
            
        with tf.variable_scope("fully_connected2") as scope:
            net, weights_fc2 = tf_util_angmargin.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
            
        with tf.variable_scope("dropout2") as scope:
            net = tf_util_angmargin.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
        
        with tf.variable_scope("fully_connected3") as scope:
            # net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')        # original
            net, weights_fc3 = tf_util_angmargin.fully_connected(net, num_class, activation_fn=None, scope='fc3')   # Bernardo

        # Add one more layer to implement angular margin
        with tf.variable_scope("fully_connected4") as scope:
            net, weights_fc4 = tf_util_angmargin.fully_connected(net, num_class, activation_fn=None, scope='fc4')   # Bernardo

    # return net, end_points, weights_fc3
    return net, end_points, weights_fc4



# Bernardo
# from: https://github.com/luckycallor/InsightFace-tensorflow/blob/0fda5dc7fe2a651de08b0ed1bb7cc0ebc2dcd9f7/losses/logit_loss.py#L21
def get_loss_arcface(embd, labels, end_points, weights_fc, num_classes, m=0.5, s=32.0):
    embds = tf.nn.l2_normalize(embd, dim=1, name='normed_embd')
    weights = tf.nn.l2_normalize(weights_fc, dim=0)

    cos_m = math.cos(m)
    sin_m = math.sin(m)

    mm = sin_m * m

    threshold = math.cos(math.pi - m)

    cos_t = tf.matmul(embds, weights, name='cos_t')

    cos_t2 = tf.square(cos_t, name='cos_2')
    sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
    sin_t = tf.sqrt(sin_t2, name='sin_t')
    cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
    cond_v = cos_t - threshold
    cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
    keep_val = s*(cos_t - mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)
    mask = tf.one_hot(labels, depth=num_classes, name='one_hot_mask')
    inv_mask = tf.subtract(1., mask, name='inverse_mask')
    s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
    logits = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    
    return logits, loss, classify_loss


# Bernardo
def get_loss_common_cross_entropy(embd, label, end_points, weights_fc, num_classes, m=0.5, s=30.0):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=embd, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return embd, loss, classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        num_classes = 5
        num_points = 1024

        pointclouds_pl, labels_pl = placeholder_inputs(batch_size=8, num_point=num_points)

        is_training_pl = tf.placeholder(tf.bool, shape=())

        pred, end_points, weights_fc3 = get_model(pointclouds_pl, is_training_pl, bn_decay=None, num_class=num_classes)    # Bernardo   
        # loss, classify_loss, classify_test_loss, selected_labels, cosine_sim, one_hot_labels = get_loss(pred, labels_pl, end_points, weights_fc3, num_classes, s=30.0, m=0.5)
        logits, loss, classify_loss = get_loss(pred, labels_pl, end_points, weights_fc3, num_classes, s=30.0, m=0.5)

        # data1 = np.ones((32,1024,3),dtype=np.float32)
        data = np.random.rand(8, num_points, 3)
        
        # true_labels = np.zeros((8,),dtype=np.int32)
        true_labels = np.random.randint(num_classes, size=8)
        
        is_training = False
        sess = tf.Session()
        with sess.as_default():
            print_op = tf.Print(pointclouds_pl, [logits])
            with tf.control_dependencies([print_op]):
                out = tf.add(logits, logits)
            sess.run(tf.global_variables_initializer())
            
            logits, loss, classify_loss = sess.run([logits, loss, classify_loss], feed_dict={pointclouds_pl: data,
                                                                labels_pl: true_labels,
                                                                is_training_pl: is_training})

            print('logits:', logits)
            print('loss:', loss)
            print('classify_loss:', classify_loss)
            
            