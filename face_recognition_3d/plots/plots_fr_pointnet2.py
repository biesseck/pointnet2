from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import sys
import os
import numpy as np
from matplotlib import pyplot
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_path', type=str, default='/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-24_SyntheticFaces_dataset_100classes_50exp_lr=0.001_batch=64/log_train.txt', help='Path of train log file')

    return parser.parse_args()


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


def load_original_training_log_pointnet2_verif_pairs(path_file=''):
    with open(path_file) as f:
        all_lines = f.readlines()
        parameters, epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy = [], [], [], [], [], []
        for i, line in enumerate(all_lines):
            line = line[:-1]
            # print('line:', line)
            if line.startswith('Namespace'):
                # parameters = line.replace('Namespace', '').replace('(', '').replace(')', '')
                parameters = line.replace('Namespace', '')
                # print('parameters:', parameters)

            if line.startswith('---- EPOCH') and line.endswith('TRAIN EVALUATION ----'):
                epoch.append(int(line.split(' ')[2]))
                if i+2 < len(all_lines) and all_lines[i+2].startswith('train mean loss:'):
                    train_mean_loss.append(float(all_lines[i+2][:-1].split(':')[-1]))
                if i+3 < len(all_lines) and all_lines[i+3].startswith('train accuracy'):
                    train_accuracy.append(float(all_lines[i+3][:-1].split(':')[-1]))

            if line.startswith('---- EPOCH') and line.endswith('TEST EVALUATION ----'):
                if i+2 < len(all_lines) and all_lines[i+2].startswith('test mean loss:'):
                    test_mean_loss.append(float(all_lines[i+2][:-1].split(':')[-1]))
                if i+3 < len(all_lines) and all_lines[i+3].startswith('test accuracy'):
                    test_accuracy.append(float(all_lines[i+3][:-1].split(':')[-1]))

        epoch = np.array(epoch, dtype=np.int32)
        train_mean_loss = np.array(train_mean_loss, dtype=np.float32)
        train_accuracy = np.array(train_accuracy, dtype=np.float32)
        test_mean_loss = np.array(test_mean_loss, dtype=np.float32)
        test_accuracy = np.array(test_accuracy, dtype=np.float32)

        # print('epoch:', epoch)
        # print('train_mean_loss:', train_mean_loss)
        # print('train_accuracy:', train_accuracy)
        # print('test_mean_loss:', test_mean_loss)
        # print('test_accuracy:', test_accuracy)

        return parameters, epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy


def load_original_training_log_pointnet2_angmargin(path_file=''):
    with open(path_file) as f:
        all_lines = f.readlines()
        parameters, epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy = [], [], [], [], [], []
        for i, line in enumerate(all_lines):
            line = line[:-1]
            # print('line:', line)
            if line.startswith('Namespace'):
                # parameters = line.replace('Namespace', '').replace('(', '').replace(')', '')
                parameters = line.replace('Namespace', '')
                # print('parameters:', parameters)

            if line.startswith('---- EPOCH') and line.endswith('TRAIN EVALUATION ----'):
                epoch.append(int(line.split(' ')[2]))
                if i+2 < len(all_lines) and all_lines[i+2].startswith('train mean loss:'):
                    train_mean_loss.append(float(all_lines[i+2][:-1].split(':')[-1]))
                if i+3 < len(all_lines) and all_lines[i+3].startswith('train accuracy'):
                    train_accuracy.append(float(all_lines[i+3][:-1].split(':')[-1]))

            if line.startswith('---- EPOCH') and line.endswith('TEST EVALUATION ----'):
                if i+2 < len(all_lines) and all_lines[i+2].startswith('test mean loss:'):
                    test_mean_loss.append(float(all_lines[i+2][:-1].split(':')[-1]))
                if i+3 < len(all_lines) and all_lines[i+3].startswith('test accuracy'):
                    test_accuracy.append(float(all_lines[i+3][:-1].split(':')[-1]))

        max_loss_value = max(max(train_mean_loss), max(test_mean_loss))
        epoch = np.array(epoch, dtype=np.int32)
        train_mean_loss = np.array(train_mean_loss, dtype=np.float32) / max_loss_value
        train_accuracy = np.array(train_accuracy, dtype=np.float32)
        test_mean_loss = np.array(test_mean_loss, dtype=np.float32) / max_loss_value
        test_accuracy = np.array(test_accuracy, dtype=np.float32)

        # print('epoch:', epoch)
        # print('train_mean_loss:', train_mean_loss)
        # print('train_accuracy:', train_accuracy)
        # print('test_mean_loss:', test_mean_loss)
        # print('test_accuracy:', test_accuracy)

        return parameters, epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy

                                    
def plot_training_history_pointnet2(epoch, eval_mean_loss, eval_accuracy, eval_avg_class_acc, title='', subtitle='', path_image='.', show_fig=False, save_fig=False):
    # plot loss during training
    pyplot.clf()

    pyplot.subplot(211)
    pyplot.suptitle(title, fontsize=11, fontweight='bold')
    pyplot.title(subtitle, fontsize=8)
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    pyplot.plot(epoch, eval_mean_loss, label='eval_mean_loss', color='red')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Error')
    pyplot.ylim(0, np.nanmax(eval_mean_loss)*1.25)
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

    pyplot.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.75, wspace=0.4, hspace=0.4)
    
    if save_fig:
        pyplot.savefig(path_image)
        if path_image.endswith('.png'):
            pyplot.savefig(path_image.replace('.png', '.svg'))
        if path_image.endswith('.jpg'):
            pyplot.savefig(path_image.replace('.jpg', '.svg'))
    if show_fig:
        pyplot.show()


def plot_training_history_pointnet2_verif_pairs(epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy, title='', subtitle='', path_image='.', show_fig=False, save_fig=False):
    # plot loss during training
    pyplot.clf()
    # eval_accuracy = eval_mean_loss.copy()
    if len(train_mean_loss) > 0:
        if len(test_accuracy) > 0:
            pyplot.subplot(211)
        pyplot.plot(epoch, train_mean_loss, label='train_mean_loss', color='red')
        pyplot.plot(epoch, test_mean_loss, label='test_mean_loss', color='blue')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Error')
        pyplot.ylim(0, np.nanmax(train_mean_loss)*1.25)
        pyplot.legend()
    
    pyplot.suptitle(title, fontsize=11, fontweight='bold')
    pyplot.title(subtitle, fontsize=8)
    
    if len(test_accuracy) > 0:
        # plot accuracy during training
        pyplot.subplot(212)
        # pyplot.title('Accuracy')
        pyplot.plot(epoch, train_accuracy, label='train_accuracy', color='red')
        pyplot.plot(epoch, test_accuracy, label='test_accuracy', color='blue')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Accuracy')
        pyplot.ylim(0, 1)
        pyplot.legend()

    pyplot.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.75, wspace=0.4, hspace=0.4)

    if save_fig:
        pyplot.savefig(path_image)
        if path_image.endswith('.png'):
            pyplot.savefig(path_image.replace('.png', '.svg'))
        if path_image.endswith('.jpg'):
            pyplot.savefig(path_image.replace('.jpg', '.svg'))
    if show_fig:
        pyplot.show()


def plot_training_history_pointnet2_angmargin(epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy, title='', subtitle='', path_image='.', show_fig=False, save_fig=False):
    # plot loss during training
    pyplot.clf()
    # eval_accuracy = eval_mean_loss.copy()
    if len(train_mean_loss) > 0:
        if len(test_accuracy) > 0:
            pyplot.subplot(211)
        pyplot.plot(epoch, train_mean_loss, label='train_mean_loss', color='red')
        pyplot.plot(epoch, test_mean_loss, label='test_mean_loss', color='blue')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Error')
        # pyplot.ylim(0, np.nanmax(train_mean_loss)*1.25)
        pyplot.ylim(0, 1)
        pyplot.legend()
    
    pyplot.suptitle(title, fontsize=11, fontweight='bold')
    pyplot.title(subtitle, fontsize=8)
    
    if len(test_accuracy) > 0:
        # plot accuracy during training
        pyplot.subplot(212)
        # pyplot.title('Accuracy')
        pyplot.plot(epoch, train_accuracy, label='train_accuracy', color='red')
        pyplot.plot(epoch, test_accuracy, label='test_accuracy', color='blue')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Accuracy')
        pyplot.ylim(0, 1)
        pyplot.legend()

    pyplot.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.75, wspace=0.4, hspace=0.4)

    if save_fig:
        pyplot.savefig(path_image)
        if path_image.endswith('.png'):
            pyplot.savefig(path_image.replace('.png', '.svg'))
        if path_image.endswith('.jpg'):
            pyplot.savefig(path_image.replace('.jpg', '.svg'))
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



def plot_samples_per_class_and_histogram(class_names, samples_per_class, log_scale=False, title='', subtitle='', path_image='.', show_fig=False, save_fig=False):
    pyplot.clf()

    sorted_class_by_samples = sorted(zip(samples_per_class, class_names), reverse=True)
    samples_per_class = np.array([num_samples for num_samples, _ in sorted_class_by_samples], dtype=np.int)
    class_names       = [class_name  for _, class_name  in sorted_class_by_samples]
    class_indexes     = range(len(samples_per_class))

    x_divisions = 7
    x_block = int(round(len(class_indexes)/x_divisions))
    sampled_class_indexes = [class_indexes[i] for i in class_indexes if i%x_block==0]
    sampled_class_names = [class_names[i] for i in class_indexes if i%x_block==0]
    # for num_samples, class_name in sorted_class_by_samples:
    #     print('class_name:', class_name, '    num_samples:', num_samples)
    
    if title != '':
        pyplot.suptitle(title, fontsize=12, fontweight='bold')
    
    pyplot.subplot(211)
    if subtitle != '':
        pyplot.title(subtitle, fontsize=8)
    pyplot.plot(class_indexes, samples_per_class, color='red')
    pyplot.xticks(sampled_class_indexes, sampled_class_names, rotation=-15, fontsize=7, ha='left')
    ylabel = '# Samples'
    if log_scale:
        pyplot.yscale('log')
        ylabel += ' (log scale)'
    pyplot.ylabel(ylabel)
    # pyplot.ylim(0, np.nanmax(samples_per_class)*1.25)
    pyplot.xlabel('Subjects')
    # pyplot.xlim(len(class_names)*-0.25, len(class_names)*1.25)
    pyplot.legend()
    
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.hist(samples_per_class, bins=20, color='blue')
    # pyplot.yscale('log')
    ylabel = '# classes'
    if log_scale:
        pyplot.yscale('log')
        ylabel += ' (log scale)'
    pyplot.ylabel(ylabel)
    pyplot.xlabel('Num samples per subject')
    pyplot.legend()

    pyplot.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.7)
    
    if save_fig:
        pyplot.savefig(path_image)
    if show_fig:
        pyplot.show()
 


if __name__ == '__main__':

    if not '-input_path' in sys.argv:
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.001[DEFAULT]/log_train.txt']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.001[DEFAULT]/log_train.txt']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.001/log_train.txt']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.001_normaliz=min-max/log_train.txt']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.001_normaliz=min-max_div=100/log_train.txt']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.0001_normaliz=min-max_div=100/log_train.txt']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.005/log_train.txt']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-21_FGRCv2_dataset_133classes_lr=0.01/log_train.txt']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/log_face_recognition_2022-10-24_SyntheticFaces_dataset_100classes_10exp_lr=0.001_batch=32/log_train.txt']
        sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/classification/log_face_recognition_train_arcface=ms1mv2-reduced_batch=16_margin=0.0_TEST-PLOT/log_train.txt']


    args = parse_args()

    # load_original_training_log_pointnet2(path_file=args.input_path)

    # parameters, epoch, eval_mean_loss, eval_accuracy, eval_avg_class_acc = load_original_training_log_pointnet2(path_file=args.input_path)
    # parameters, epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy = load_original_training_log_pointnet2_verif_pairs(path_file=args.input_path)
    parameters, epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy = load_original_training_log_pointnet2_angmargin(path_file=args.input_path)
    
    title = 'PointNet++ training on MS1MV2_reduced-Reconst3D-MICA \nClassification (1:N) - '+str(22)+' classes - min_samples='+str(3)+' - max_samples='+str(-1)
    subtitle = 'Parameters: ' + break_string(parameters, substring=', ', num_parts=4)
    path_image = '/'.join(args.input_path.split('/')[:-1]) + '/training_history_from_log_file.png'
    print('Saving training history:', path_image)
    plot_training_history_pointnet2_angmargin(epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy, title=title, subtitle=subtitle, path_image=path_image, show_fig=False, save_fig=True)

    