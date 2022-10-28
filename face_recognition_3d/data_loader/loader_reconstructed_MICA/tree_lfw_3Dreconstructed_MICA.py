from __future__ import print_function

import sys
import os
from glob import glob
from pathlib import Path

# print('os.path.dirname(os.path.abspath(__file__)):', os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from plots import plots_fr_pointnet2


# BERNARDO
class TreeLFW_3DReconstructedMICA:
    '''
    def __walk(self, dir_path=Path()):
        contents = list(dir_path.iterdir())
        for path in contents:
            if path.is_dir():  # extend the prefix and recurse:
                yield str(path)
                # yield from self.__walk(path)
                yield self.__walk(path)
    '''

    def get_all_sub_folders(self, dir_path=''):
        return sorted(glob(dir_path + '/*/*/'))

    def get_sub_folders_one_level(self, dir_path=''):
        # sub_folders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        sub_folders = [f.name for f in os.scandir(dir_path) if f.is_dir()]
        return sorted(sub_folders)

    def get_all_pointclouds_paths(self, dir_path, pc_ext='.ply'):
        all_sub_folders = self.get_all_sub_folders(dir_path)
        all_pc_paths = []
        all_pc_subjects = []
        for sub_folder_pointcloud in all_sub_folders:
            pc_paths = sorted(glob(sub_folder_pointcloud + '/*' + pc_ext))
            # print('pc_paths:', pc_paths)
            pc_subjects = [pc_path.split('/')[-3] for pc_path in pc_paths]
            all_pc_paths += pc_paths
            all_pc_subjects += pc_subjects
        return all_pc_paths, all_pc_subjects

    def count_samples_per_subject(self, pc_paths_list=[''], pc_ext='.ply'):
        unique_subjects_names = []
        samples_per_subject = []
        for pc_path in sorted(pc_paths_list):
            data_path = pc_path.split('/')
            if pc_ext in data_path[-1]:
                subject_name = data_path[-3]
                if not subject_name in unique_subjects_names:
                    samples_per_subject.append(0)
                    unique_subjects_names.append(subject_name)
                samples_per_subject[-1] += 1
        assert len(unique_subjects_names) == len(samples_per_subject)
        return unique_subjects_names, samples_per_subject

    def get_all_pointclouds_paths_count(self, dir_path, pc_ext='.ply'):
        all_pc_paths, all_pc_subjects = self.get_all_pointclouds_paths(dir_path, pc_ext)
        unique_subjects_names, samples_per_subject = self.count_samples_per_subject(all_pc_paths, pc_ext)
        return all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject

    def filter_paths_by_minimum_samples(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, pc_ext='.ply', min_samples=2, max_samples=-1):
        filtered_pc_paths = []
        filtered_pc_subjects = []
        filtered_subjects_names = []
        filtered_samples_per_subject = []
        selected_samples_per_subject = [0] * len(unique_subjects_names)
        for i, pc_path, pc_subj in zip(range(len(all_pc_paths)), all_pc_paths, all_pc_subjects):
            if samples_per_subject[unique_subjects_names.index(pc_subj)] >= min_samples and \
               (max_samples==-1 or selected_samples_per_subject[unique_subjects_names.index(pc_subj)] < max_samples):
                filtered_pc_paths.append(pc_path)
                filtered_pc_subjects.append(pc_subj)
                if not pc_subj in filtered_subjects_names:   # run once per subject
                    filtered_subjects_names.append(pc_subj)
                selected_samples_per_subject[unique_subjects_names.index(pc_subj)] += 1
        # filtered_samples_per_subject.append(samples_per_subject[unique_subjects_names.index(pc_subj)])
        filtered_samples_per_subject = [selected_samples_per_subject[unique_subjects_names.index(pc_subj)] for pc_subj in filtered_subjects_names]
        # print('selected_samples_per_subject:', selected_samples_per_subject)      
        return filtered_pc_paths, filtered_pc_subjects, filtered_subjects_names, filtered_samples_per_subject

    def load_filter_organize_pointclouds_paths(self, dir_path, pc_ext='.ply', min_samples=2, max_samples=-1):
        all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = self.get_all_pointclouds_paths_count(dir_path, pc_ext)
        all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = self.filter_paths_by_minimum_samples(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, pc_ext, min_samples, max_samples)
        subjects_with_pc_paths = [()] * len(all_pc_paths)
        for i, pc_path, pc_subj in zip(range(len(all_pc_paths)), all_pc_paths, all_pc_subjects):
            subjects_with_pc_paths[i] = (pc_subj, pc_path)
        # print('samples_per_subject:', samples_per_subject)
        return subjects_with_pc_paths, unique_subjects_names, samples_per_subject




if __name__ == '__main__':
    dataset_path = '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw'
    
    file_ext='.ply'
    
    min_samples=1
    # min_samples=3

    # max_samples=-1
    max_samples=100

    log_scale = True
    # log_scale = False

    # all_pc_paths, all_pc_subjects = TreeLFW_3DReconstructedMICA().get_all_pointclouds_paths(dir_path=dataset_path, pc_ext=file_ext)
    # unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().count_samples_per_subject(all_pc_paths, pc_ext=file_ext)
    # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # for i, unique_subj_name in enumerate(unique_subjects_names):
    #     print('unique_subj_name', unique_subj_name, '    samples_per_subject:', samples_per_subject[i])
    # sys.exit(0)
    
    # print('Searching all files ending with \'' + file_ext + '\'...')
    # all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().get_all_pointclouds_paths_count(dataset_path, file_ext)
    # # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    # #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # # sys.exit(0)
    # # for unique_subj_name, samples_per_subj in zip(unique_subjects_names, samples_per_subject):
    # #     print('unique_subj_name:', unique_subj_name, '    samples_per_subj:', samples_per_subj)
    # # sys.exit(0)

    # print('Searching all files ending with \'' + file_ext + '\'...')
    # all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().filter_paths_by_minimum_samples(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, file_ext, min_samples)
    # # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    # #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # # sys.exit(0)
    # # for unique_subj_name, samples_per_subj in zip(unique_subjects_names, samples_per_subject):
    # #     print('unique_subj_name:', unique_subj_name, '    samples_per_subj:', samples_per_subj)
    # # print('len(unique_subj_name):', len(unique_subjects_names), '    len(samples_per_subj):', len(samples_per_subject))
    # # sys.exit(0)
    
    print('Searching all files ending with \'' + file_ext + '\'...')
    subjects_with_pc_paths, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().load_filter_organize_pointclouds_paths(dataset_path, file_ext, min_samples, max_samples)
    # for subj_pc_path in subjects_with_pc_paths:
    #     print('subj_pc_path:', subj_pc_path)
    # # sys.exit(0)

    # unique_subjects_names = ['AAAAA', 'BBBB', 'CCCC', 'DDDD']
    # samples_per_subject = [5, 2, 2, 1]
    title = 'Dataset LFW - Samples per Subject'
    subtitle = '(min_samples='+str(min_samples)+', max_samples='+str(max_samples)+')'
    path_image = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/samples_per_subject_lfw_dataset_minsamples='+str(min_samples)+'_maxsamples='+str(max_samples)+'.png'
    plots_fr_pointnet2.plot_samples_per_class_and_histogram(unique_subjects_names, samples_per_subject, log_scale, title=title, subtitle=subtitle, path_image=path_image, show_fig=False, save_fig=True)
    
