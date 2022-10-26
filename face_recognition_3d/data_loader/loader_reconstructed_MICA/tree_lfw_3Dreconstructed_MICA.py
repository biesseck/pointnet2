from __future__ import print_function

import sys
import os
from glob import glob
from pathlib import Path

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

    def filter_paths_by_minimum_samples(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, pc_ext='.ply', min_samples=2):
        filtered_pc_paths = []
        filtered_pc_subjects = []
        filtered_subjects_names = []
        filtered_samples_per_subject = []
        # for i, pc_path, pc_subj in enumerate(all_pc_paths, all_pc_subjects):
        for i, pc_path, pc_subj in zip(range(len(all_pc_paths)), all_pc_paths, all_pc_subjects):
            if samples_per_subject[unique_subjects_names.index(pc_subj)] >= min_samples:
                filtered_pc_paths.append(pc_path)
                filtered_pc_subjects.append(pc_subj)
                if not pc_subj in filtered_subjects_names:
                    filtered_subjects_names.append(pc_subj)
                    filtered_samples_per_subject.append(samples_per_subject[unique_subjects_names.index(pc_subj)])
        return filtered_pc_paths, filtered_pc_subjects, filtered_subjects_names, filtered_samples_per_subject



    '''
    def get_pointclouds_paths(self, dir_path, num_classes=1000, num_expressions=50, pc_ext='.ply'):
        all_sub_folders = self.get_all_sub_folders(dir_path)[1:]
        pc_paths = [[]] * num_classes
        for i in range(num_classes):
            sub_folder_pointcloud = all_sub_folders[i]
            paths_one_folder = sorted(glob.glob(sub_folder_pointcloud + '/*' + pc_ext))
            pc_paths[i] = paths_one_folder[:num_expressions]
        return pc_paths

    def get_pointclouds_paths_with_subjects_names(self, dir_path, num_classes=1000, num_expressions=50, pc_ext='.ply'):
        assert num_classes > 0 and num_classes <= 10000 and num_expressions > 0 and num_expressions <= 50
        all_sub_folders = self.get_all_sub_folders(dir_path)[1:]
        pc_paths = [[]] * num_classes
        subjects_names = [[]] * num_classes
        unique_subjects_names = [''] * num_classes
        for i in range(num_classes):
            sub_folder_pointcloud = all_sub_folders[i]
            paths_one_folder = sorted(sorted(glob.glob(sub_folder_pointcloud + '/*' + pc_ext)))
            pc_paths[i] = paths_one_folder[:num_expressions]
            subject_name = paths_one_folder[0].split('/')[-2]
            subjects_names[i] = [subject_name] * num_expressions
            unique_subjects_names[i] = subject_name
        
        pc_subjects_paths = []
        for i in range(num_classes):
            for j in range(len(pc_paths[i])):
                pc_subjects_paths.append((subjects_names[i][j], pc_paths[i][j]))
        return pc_subjects_paths, unique_subjects_names
    '''



if __name__ == '__main__':
    dataset_path = '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw'
    file_ext='.ply'
    min_samples=3

    # all_pc_paths, all_pc_subjects = TreeLFW_3DReconstructedMICA().get_all_pointclouds_paths(dir_path=dataset_path, pc_ext=file_ext)
    # unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().count_samples_per_subject(all_pc_paths, pc_ext=file_ext)
    # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # for i, unique_subj_name in enumerate(unique_subjects_names):
    #     print('unique_subj_name', unique_subj_name, '    samples_per_subject:', samples_per_subject[i])
    # sys.exit(0)

    all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().get_all_pointclouds_paths_count(dataset_path, file_ext)
    # # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    # #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # # sys.exit(0)
    # # for unique_subj_name, samples_per_subj in zip(unique_subjects_names, samples_per_subject):
    # #     print('unique_subj_name:', unique_subj_name, '    samples_per_subj:', samples_per_subj)
    # # sys.exit(0)

    all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().filter_paths_by_minimum_samples(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, file_ext, min_samples)
    for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
        print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    sys.exit(0)
    # for unique_subj_name, samples_per_subj in zip(unique_subjects_names, samples_per_subject):
    #     print('unique_subj_name:', unique_subj_name, '    samples_per_subj:', samples_per_subj)
    # print('len(unique_subj_name):', len(unique_subjects_names), '    len(samples_per_subj):', len(samples_per_subject))
    # sys.exit(0)

    # pc_subjects_paths, unique_subjects_names = TreeSyntheticFacesGPMM().get_pointclouds_paths_with_subjects_names(dir_path=synthetic_faces_path, num_classes=10, num_expressions=5)
    # print('pc_subjects_paths:', pc_subjects_paths)
    # # for i in range(len(pc_subjects_paths)):
    # #     print('pc_subjects_paths[i]:', pc_subjects_paths[i])
    # print('unique_subjects_names:', unique_subjects_names)
