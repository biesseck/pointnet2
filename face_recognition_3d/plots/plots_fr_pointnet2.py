from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import sys
import os
import numpy as np
from matplotlib import pyplot


def load_original_training_log_pointnet2(path_file=''):
    with open(path_file) as f:
        all_lines = f.readlines()
        parameters, epoch, eval_mean_loss, eval_accuracy, eval_avg_class_acc = [], [], [], [], []
        for i, line in enumerate(all_lines):
            line = line[:-1]
            # print('line:', line)
            if line.startswith('Namespace'):
                # parameters = line.replace('Namespace', '').replace('(', '').replace(')', '')
                parameters = line.replace('Namespace', '')
                # print('parameters:', parameters)
            if line.startswith('---- EPOCH') and line.endswith('EVALUATION ----'):
                epoch.append(int(line.split(' ')[2]))
                eval_mean_loss.append(float(all_lines[i+1][:-1].split(':')[-1]))
                eval_accuracy.append(float(all_lines[i+2][:-1].split(':')[-1]))
                eval_avg_class_acc.append(float(all_lines[i+3][:-1].split(':')[-1]))
                # print('epoch:', epoch)
                # print('    eval_mean_loss:', eval_mean_loss)
                # print('    eval_accuracy:', eval_accuracy)
                # print('    eval_avg_class_acc:', eval_avg_class_acc)
        epoch = np.array(epoch, dtype=np.int32)
        eval_mean_loss = np.array(eval_mean_loss, dtype=np.float32)
        eval_accuracy = np.array(eval_accuracy, dtype=np.float32)
        eval_avg_class_acc = np.array(eval_avg_class_acc, dtype=np.float32)
        return parameters, epoch, eval_mean_loss, eval_accuracy, eval_avg_class_acc


def plot_training_history_pointnet2(epoch, eval_mean_loss, eval_accuracy, eval_avg_class_acc, title='', subtitle='', path_image='.', show_fig=False, save_fig=False):
    # plot loss during training
    pyplot.clf()

    pyplot.subplot(211)
    pyplot.suptitle(title)
    pyplot.title(subtitle, fontsize=8)
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    pyplot.plot(epoch, eval_mean_loss, label='eval_mean_loss', color='red')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Error')
    pyplot.ylim(0, np.max(eval_mean_loss)+np.max(eval_mean_loss)*0.25)
    pyplot.legend()
    
    # plot accuracy during training
    pyplot.subplot(212)
    # pyplot.title('Accuracy')
    pyplot.plot(epoch, eval_accuracy, label='eval_accuracy', color='blue')
    pyplot.plot(epoch, eval_avg_class_acc, label='eval_avg_class_acc', color='green')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.ylim(0, 1)
    pyplot.legend()

    pyplot.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8, wspace=0.4, hspace=0.4)
    
    if save_fig:
        pyplot.savefig(path_image)
    if show_fig:
        pyplot.show()


def break_string(text, substring=', ', num_parts=3):
    values = text.split(substring)
    size_part = int(round(len(values) / num_parts))
    if len(values) > 1:
        final_string = ''
        for i in range(len(values)):
            if not 'log_dir=' in values[i]:
                final_string += values[i]
                if i < len(values)-1:
                    final_string += ', '
                # if i == int(round(len(values)/2)):
                if i > 0 and i % size_part == 0:
                    final_string += '\n'
    else:
        final_string = text
    # print('final_string:', final_string)
    return final_string




if __name__ == '__main__':
    # path_log_file = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.001[DEFAULT]/log_train.txt'
    # path_log_file = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.001/log_train.txt'
    # path_log_file = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.001_normaliz=min-max/log_train.txt'
    # path_log_file = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.001_normaliz=min-max_div=100/log_train.txt'
    # path_log_file = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.0001_normaliz=min-max_div=100/log_train.txt'
    # path_log_file = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.005/log_train.txt'
    # path_log_file = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.01/log_train.txt'
    # path_log_file = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-24_SyntheticFaces_dataset_100classes_10exp_lr=0.001_batch=32/log_train.txt'
    path_log_file = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-24_SyntheticFaces_dataset_100classes_50exp_lr=0.001_batch=64/log_train.txt'


    # load_original_training_log_pointnet2(path_file=path_log_file)

    parameters, epoch, eval_mean_loss, eval_accuracy, eval_avg_class_acc = load_original_training_log_pointnet2(path_file=path_log_file)
    
    title = 'PointNet++ training on SyntheticFaces (100 classes) - Classification (1:N)'
    subtitle = 'Parameters: ' + break_string(parameters, substring=', ')
    # path_image = './training_history.png'
    path_image = '/'.join(path_log_file.split('/')[:-1]) + '/training_history_from_log_file.png'
    plot_training_history_pointnet2(epoch, eval_mean_loss, eval_accuracy, eval_avg_class_acc, title=title, subtitle=subtitle, path_image=path_image, show_fig=False, save_fig=True)

    