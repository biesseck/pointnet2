'''
    Evaluate classification performance with optional voting.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../models'))
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import provider
# import modelnet_dataset
# import modelnet_h5_dataset

from data_loader.loader_reconstructed_MICA import lfw_3Dreconstructed_MICA_dataset_pairs     # Bernardo

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name [default: pointnet2_cls_ssg]')  # original
parser.add_argument('--model', default='pointnet2_verif_pairs_ssg', help='Model name [default: pointnet2_verif_pairs_ssg]')    # Bernardo
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')    # original
parser.add_argument('--num_point', type=int, default=2900, help='Point Number [default: 1024]')      # Bernardo
# parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')   # original
parser.add_argument('--model_path', default='logs_training/verification/log_face_recognition_train=ms1mv2_subj=10408_pairs=100000_lrate=0.00005_batch=8/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
# parser.add_argument('--normal', action='store_true', help='Whether to use normal information')      # original
parser.add_argument('--normal', type=bool, default=False, help='Whether to use normal information')   # Bernardo
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
parser.add_argument('--margin', type=float, default=0.5, help='Minimum distance for non-corresponding pairs in Contrastive Loss')

# parser.add_argument('--dataset', type=str, default='frgc', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='synthetic_gpmm', help='Name of dataset to train model')   # Bernardo
parser.add_argument('--dataset', type=str, default='reconst_mica_lfw', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2', help='Name of dataset to train model')   # Bernardo

FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MARGIN = FLAGS.margin

MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()


if FLAGS.dataset.upper() == 'reconst_mica_lfw'.upper():
    DATA_PATH = os.path.join(ROOT_DIR, '../../MICA/demo/output/lfw')
    TRAIN_DATASET = lfw_3Dreconstructed_MICA_dataset_pairs.LFR_3D_Reconstructed_MICA_Dataset_Pairs(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET  = lfw_3Dreconstructed_MICA_dataset_pairs.LFR_3D_Reconstructed_MICA_Dataset_Pairs(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        # pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        pointclouds_pl1, pointclouds_pl2, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        # pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        pred1, end_points1, pred2, end_points2 = MODEL.get_model(pointclouds_pl1, pointclouds_pl2, is_training_pl, bn_decay=None)    # Bernardo
            
        # MODEL.get_loss(pred, labels_pl, end_points)
        _, individual_losses, distances, pred_labels = MODEL.get_loss(pred1, pred2, labels_pl, end_points1, end_points2, MARGIN)

        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl1': pointclouds_pl1,
               'pointclouds_pl2': pointclouds_pl2,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'individual_losses': individual_losses,
               'total_loss': total_loss,
               'pred1': pred1,
               'pred2': pred2,
               'distances': distances,
               'pred_labels': pred_labels,
               'end_points1': end_points1,
               'end_points2': end_points2}

    eval_one_epoch(sess, ops, num_votes)

def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False

    # Make sure batch data is of same size
    # cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    # cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_batch_data = np.zeros((2,BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    # total_seen_class = [0 for _ in range(NUM_CLASSES)]
    # total_correct_class = [0 for _ in range(NUM_CLASSES)]

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        # bsize = batch_data.shape[0]   # original
        bsize = batch_data.shape[1]    # Bernardo
        
        print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        
        # cur_batch_data[0:bsize,...] = batch_data
        # cur_batch_label[0:bsize] = batch_label
        cur_batch_data[0,0:bsize,...] = batch_data[0]
        cur_batch_data[1,0:bsize,...] = batch_data[1]
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl1']: cur_batch_data[0],
                     ops['pointclouds_pl2']: cur_batch_data[1],
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}

        # loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
        loss_val, ind_loss, pred_labels = sess.run([ops['total_loss'], ops['individual_losses'], ops['pred_labels']], feed_dict=feed_dict)
        # batch_pred_sum += pred_labels

        print('pred_labels:', pred_labels[0:bsize])
        print('batch_label:', batch_label[0:bsize])
        print('bsize:', bsize)
        
        # pred_val = np.argmax(batch_pred_sum, 1)
        # pred_labels = pred_labels[0]
        correct = np.sum(np.array(pred_labels[0:bsize], dtype=np.int) == np.array(batch_label[0:bsize], dtype=np.int))
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1

    
    test_mean_loss = loss_sum / float(batch_idx)
    test_accuracy = total_correct / float(total_seen)
    log_string('test loss sum: %f' % (loss_sum))
    log_string('test mean loss: %f' % (test_mean_loss))
    log_string('test accuracy: %f'% (test_accuracy))
    
    


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
