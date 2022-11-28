'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
from __future__ import print_function

import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, '../models'))
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import provider
import tf_util

# Bernardo
from plots import plots_fr_pointnet2

# import modelnet_dataset     # original
# import modelnet_h5_dataset  # original
from data_loader.loader_frgc2 import frgc2_dataset                                           # Bernardo
from data_loader.loader_synthetic_faces_gpmm import synthetic_faces_gpmm_dataset             # Bernardo
from data_loader.loader_reconstructed_MICA import lfw_3Dreconstructed_MICA_dataset           # Bernardo
from data_loader.loader_reconstructed_MICA import ms1mv2_3Dreconstructed_MICA_dataset        # Bernardo


# os.environ["CUDA_VISIBLE_DEVICES"]='-1'   # cpu
# os.environ["CUDA_VISIBLE_DEVICES"]='0'  # gpu
# os.environ["CUDA_VISIBLE_DEVICES"]='1'  # gpu


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg_angmargin', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--log_dir', default='log_face_recognition', help='Log dir [default: log]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')    # original
parser.add_argument('--num_point', type=int, default=2900, help='Point Number [default: 1024]')      # Bernardo
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 251]')
# parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')  # original
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')    # Bernardo
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

# parser.add_argument('--normal', action='store_true', help='Whether to use normal information')     # original
# parser.add_argument('--normal', type=bool, default=True, help='Whether to use normal information')   # Bernardo
parser.add_argument('--normal', type=bool, default=False, help='Whether to use normal information')   # Bernardo

# parser.add_argument('--dataset', type=str, default='frgc', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='synthetic_gpmm', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_lfw', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2', help='Name of dataset to train model')   # Bernardo
parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2_reduced', help='Name of dataset to train model')   # Bernardo


FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, '../models', FLAGS.model+'.py')
LOG_DIR = os.path.dirname(os.path.abspath(__file__)) + '/logs_training/classification/' + FLAGS.log_dir   # Bernardo

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_face_recognition_class.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FILE_NAME = 'log_train.txt'
LOG_FOUT = open(os.path.join(LOG_DIR, LOG_FILE_NAME), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

TRAIN_SAMPLES_FILE_NAME = 'samples_train.txt'
TEST_SAMPLES_FILE_NAME  = 'samples_test.txt'

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

BEST_MEAN_LOSS = float('inf')
BEST_ACC = float('-inf')

# NUM_CLASSES = 40    # original


# Shapenet official train/test split
# if FLAGS.normal:
#     assert(NUM_POINT<=10000)

if FLAGS.dataset.upper() == 'frgc'.upper() or FLAGS.dataset.upper() == 'frgcv2'.upper():
    DATA_PATH = os.path.join(ROOT_DIR, '../data/FRGCv2.0/FRGC-2.0-dist')
    TRAIN_DATASET = frgc2_dataset.FRGCv2_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET  = frgc2_dataset.FRGCv2_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

elif FLAGS.dataset.upper() == 'synthetic_gpmm'.upper():
    DATA_PATH = os.path.join(ROOT_DIR, '../../3DFacePointCloudNet/Data/TrainData')
    n_classes = 100
    n_expressions = 10
    TRAIN_DATASET = synthetic_faces_gpmm_dataset.SyntheticFacesGPMM_Dataset(root=DATA_PATH, npoints=NUM_POINT, num_classes=n_classes, num_expressions=n_expressions, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET  = synthetic_faces_gpmm_dataset.SyntheticFacesGPMM_Dataset(root=DATA_PATH, npoints=NUM_POINT, num_classes=n_classes, num_expressions=n_expressions, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

elif FLAGS.dataset.upper() == 'reconst_mica_lfw'.upper():
    DATA_PATH = os.path.join(ROOT_DIR, '../../MICA/demo/output/lfw')
    
    # min_samples = 3
    # min_samples = 10
    min_samples = 20
    # min_samples = 50

    # max_samples = -1
    # max_samples = 3
    # max_samples = 10
    max_samples = 20
    # max_samples = 50
    
    TRAIN_DATASET = lfw_3Dreconstructed_MICA_dataset.LFR_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET  = lfw_3Dreconstructed_MICA_dataset.LFR_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

elif FLAGS.dataset.upper() == 'reconst_mica_ms1mv2'.upper():
    min_samples, max_samples = 3, -1
    
    DATA_PATH = os.path.join(ROOT_DIR, '../../MICA/demo/output/MS-Celeb-1M/ms1m-retinaface-t1/images')
    print('Loading train data...')
    TRAIN_DATASET = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    print('Loading test data...')
    TEST_DATASET  = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

elif FLAGS.dataset.upper() == 'reconst_mica_ms1mv2_reduced'.upper():
    min_samples, max_samples = 3, -1
    DATA_PATH = os.path.join(ROOT_DIR, '../../MICA/demo/output/MS-Celeb-1M/ms1m-retinaface-t1/images_reduced')
    print('Loading train data...')
    TRAIN_DATASET = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    print('Loading test data...')
    TEST_DATASET  = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)


# Bernardo
assert TRAIN_DATASET.num_classes == TEST_DATASET.num_classes
NUM_CLASSES = TRAIN_DATASET.num_classes

def save_train_test_samples(samples_list, path_output_file):
    with open(path_output_file, 'w') as file_handler:
        for item in samples_list:
            file_handler.write("{}\n".format(item))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            # pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)                         # original
            pred, end_points, weights_fc3 = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay, num_class=NUM_CLASSES)    # Bernardo
            pred, loss, classify_loss = MODEL.get_loss(pred, labels_pl, end_points, weights_fc3, TRAIN_DATASET.num_classes)
            
            
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # print "--- Get training operator"    # original
            print("--- Get training operator")     # Bernardo (for python 3.7)
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Bernardo
            plot_classification_training_history()

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        # batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)   # original
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=False)    # Bernardo
        
        #batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        if (batch_idx+1)%50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 50))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
    
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    return total_correct/float(total_seen)


# Bernardo
def plot_classification_training_history():
    path_log_file = os.path.join(LOG_DIR, LOG_FILE_NAME)
    parameters, epoch, eval_mean_loss, eval_accuracy, eval_avg_class_acc = plots_fr_pointnet2.load_original_training_log_pointnet2(path_file=path_log_file)

    if FLAGS.dataset.upper() == 'frgc'.upper() or FLAGS.dataset.upper() == 'frgcv2'.upper():
        title = 'PointNet++ training on FRGCv2 \nClassification (1:N) - '+str(NUM_CLASSES)+' classes'
    elif FLAGS.dataset.upper() == 'synthetic_gpmm'.upper():
        title = 'PointNet++ training on SyntheticFaces \nClassification (1:N) - '+str(NUM_CLASSES)+' classes - '+str(n_expressions)+' expressions'
    elif FLAGS.dataset.upper() == 'reconst_mica_lfw'.upper():
        title = 'PointNet++ training on LFW-Reconst3D-MICA \nClassification (1:N) - '+str(NUM_CLASSES)+' classes - min_samples='+str(min_samples)+' - max_samples='+str(max_samples)
    elif FLAGS.dataset.upper() == 'reconst_mica_ms1mv2'.upper():
        title = 'PointNet++ training on MS1MV2-Reconst3D-MICA \nClassification (1:N) - '+str(NUM_CLASSES)+' classes - min_samples='+str(min_samples)+' - max_samples='+str(max_samples)
    elif FLAGS.dataset.upper() == 'reconst_mica_ms1mv2_reduced'.upper():
        title = 'PointNet++ training on MS1MV2_reduced-Reconst3D-MICA \nClassification (1:N) - '+str(NUM_CLASSES)+' classes - min_samples='+str(min_samples)+' - max_samples='+str(max_samples)

    subtitle = 'Parameters: ' + plots_fr_pointnet2.break_string(parameters, substring=', ')
    # path_image = './training_history.png'
    path_image = '/'.join(path_log_file.split('/')[:-1]) + '/training_history_from_log_file.png'
    print('Saving training history:', path_image)
    plots_fr_pointnet2.plot_training_history_pointnet2(epoch, eval_mean_loss, eval_accuracy, eval_avg_class_acc, title=title, subtitle=subtitle, path_image=path_image, show_fig=False, save_fig=True)



if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))

    # Bernardo
    save_train_test_samples(TRAIN_DATASET.datapath, os.path.join(LOG_DIR, TRAIN_SAMPLES_FILE_NAME))
    save_train_test_samples(TEST_DATASET.datapath, os.path.join(LOG_DIR, TEST_SAMPLES_FILE_NAME))

    train()
    LOG_FOUT.close()

    # Bernardo
    plot_classification_training_history()

    print('\nFinished!\n')
