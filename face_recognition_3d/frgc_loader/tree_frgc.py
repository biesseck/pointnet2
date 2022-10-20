import sys
import os
import glob
from pathlib import Path

# BERNARDO
class TreeFRGCv2:
    def __walk(self, dir_path=Path()):
        contents = list(dir_path.iterdir())
        for path in contents:
            if path.is_dir():  # extend the prefix and recurse:
                yield str(path)
                # yield from self.__walk(path)
                yield self.__walk(path)


    def get_all_sub_folders(self, dir_path=''):
        # dir_path = dir_path.replace('//', '/')
        folders = [dir_path]
        for folder in self.__walk(Path(dir_path)):
            if isinstance(folder, str):
                folders.append(folder)
        return sorted(folders)


    def get_sub_folders_one_level(self, dir_path=''):
        # sub_folders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        sub_folders = [f.name for f in os.scandir(dir_path) if f.is_dir()]
        return sorted(sub_folders)


    def get_all_images_and_pointclouds_paths(self, dir_path, img_ext='.ppm', pc_ext='.abs.gz'):
        dir_path += '/nd1'
        all_sub_folders = self.get_all_sub_folders(dir_path)[1:]
        sub_folders_containing_pointclouds = []
        for sub_folder in all_sub_folders:
            files = glob.glob(sub_folder + '/*' + pc_ext)
            if len(files) > 0:
                sub_folders_containing_pointclouds.append(sub_folder)
                # print('sub_folder:', sub_folder)
                # print('files:', files)
        all_img_paths = []
        all_pc_paths = []
        for sub_folder_pointcloud in sub_folders_containing_pointclouds:
            img_paths = sorted(glob.glob(sub_folder_pointcloud + '/*' + img_ext))
            pc_paths =  sorted(glob.glob(sub_folder_pointcloud + '/*' + pc_ext))
            
            all_img_paths += img_paths
            all_pc_paths += pc_paths
        return all_pc_paths, all_img_paths


    def get_all_images_and_pointclouds_paths_by_season(self, dir_path, img_ext='.ppm', pc_ext='.abs.gz'):
        dir_path += '/nd1'
        all_sub_folders = self.get_all_sub_folders(dir_path)[1:]
        sub_folders_containing_pointclouds = []
        for sub_folder in all_sub_folders:
            files = glob.glob(sub_folder + '/*' + pc_ext)
            if len(files) > 0:
                sub_folders_containing_pointclouds.append(sub_folder)

        pc_subjects_paths_by_season = {}
        img_subjects_paths_by_season = {}
        unique_subjects_names_by_season = {}
        for sub_folder_pointcloud in sub_folders_containing_pointclouds:
            pc_paths =  sorted(glob.glob(sub_folder_pointcloud + '/*' + pc_ext))
            img_paths = sorted(glob.glob(sub_folder_pointcloud + '/*' + img_ext))
            
            pc_subjects_paths =  [(pc_paths[i].split('/')[-1].split('.')[0].split('d')[0], pc_paths[i]) for i in range(len(pc_paths))]
            img_subjects_paths = [(img_paths[i].split('/')[-1].split('.')[0].split('d')[0], img_paths[i]) for i in range(len(img_paths))]
            only_subjects = [subject_path[0] for subject_path in pc_subjects_paths]
    
            season = sub_folder_pointcloud.split('/')[-1]
            pc_subjects_paths_by_season[season] = pc_subjects_paths
            img_subjects_paths_by_season[season] = img_subjects_paths
            unique_subjects_names_by_season[season] = sorted(list(set(only_subjects)))

        return pc_subjects_paths_by_season, img_subjects_paths_by_season, unique_subjects_names_by_season


    def get_unique_common_subjects_names(self, subjects_by_season_dict={}):
        subjects_by_season_list = []
        for season in subjects_by_season_dict.keys():
            subjects_by_season_list.append(subjects_by_season_dict[season])
        unique_common_subjects = sorted(list(set.intersection(*map(set,subjects_by_season_list))))
        return unique_common_subjects


    def filter_only_common_subjects(self, subjects_by_season_dict={}, unique_common_subjects=[]):
        filtered_subjects_paths_by_season = {}
        for season in subjects_by_season_dict.keys():
            subjects_by_season = subjects_by_season_dict[season]
            filtered_subjects = [subjects for subjects in subjects_by_season if subjects[0] in unique_common_subjects]
            # print('filtered_subjects:', filtered_subjects)
            filtered_subjects_paths_by_season[season] = filtered_subjects
        return filtered_subjects_paths_by_season




if __name__ == '__main__':
    frgc_path = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/data/FRGCv2.0/FRGC-2.0-dist'

    # all_pc_paths, all_img_paths = TreeFRGCv2().get_all_images_and_pointclouds_paths(dir_path=frgc_path)
    # print 'all_pc_paths:', all_pc_paths
    
    pc_subjects_paths_by_season, img_subjects_paths_by_season, unique_subjects_names_by_season = TreeFRGCv2().get_all_images_and_pointclouds_paths_by_season(dir_path=frgc_path)
    print 'pc_subjects_paths_by_season.keys():', pc_subjects_paths_by_season.keys()
    print 'img_subjects_paths_by_season.keys():', img_subjects_paths_by_season.keys()
    print 'unique_subjects_names_by_season:', unique_subjects_names_by_season

    print 'pc_subjects_paths_by_season[\'Spring2003range\']:', pc_subjects_paths_by_season['Spring2003range']
    print 'pc_subjects_paths_by_season[\'Fall2003range\']:', pc_subjects_paths_by_season['Fall2003range']
    print 'pc_subjects_paths_by_season[\'Spring2004range\']:', pc_subjects_paths_by_season['Spring2004range']

    print 'unique_subjects_names_by_season[\'Spring2003range\']:', unique_subjects_names_by_season['Spring2003range']
    print 'unique_subjects_names_by_season[\'Fall2003range\']:', unique_subjects_names_by_season['Fall2003range']
    print 'unique_subjects_names_by_season[\'Spring2004range\']:', unique_subjects_names_by_season['Spring2004range']
