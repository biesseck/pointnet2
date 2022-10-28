'''
    Bernardo: Synthetic Faces GPMM dataset.
'''

from __future__ import print_function

from math import ceil, floor
import os
import os.path
import json
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../../../utils'))
import provider
import struct
from plyfile import PlyData

from tree_lfw_3Dreconstructed_MICA import TreeLFW_3DReconstructedMICA

def pc_normalize(pc):
    # Bernardo
    pc /= 100
    pc = (pc - pc.min()) / (pc.max() - pc.min())

    # l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m

    return pc

class LFR_3D_Reconstructed_MICA_Dataset():
    def __init__(self, root, batch_size = 32, npoints = 1024, min_samples=3, split='train', normalize=True, normal_channel=False, modelnet10=False, cache_size=15000, shuffle=None):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize

        # Bernardo
        # file_ext = '.ply'
        file_ext = '_centralized-nosetip_with-normals_filter-radius=100.npy'
        # file_ext = '_upsample_MetaPU_centralized-nosetip_with-normals_filter-radius=100.npy'
        subjects_with_pc_paths, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().load_filter_organize_pointclouds_paths(self.root, file_ext, min_samples)
        assert len(unique_subjects_names) == len(samples_per_subject)
        # for subj_pc_path in subjects_with_pc_paths:
        #     print('subj_pc_path:', subj_pc_path)
        # sys.exit(0)

        self.cat = unique_subjects_names    # Bernardo
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.num_classes = len(unique_subjects_names)
        self.normal_channel = normal_channel
        # print 'self.cat:', self.cat
        # print 'self.classes:', self.classes

        # Bernardo
        assert(split=='train' or split=='test')
        self.datapath = []

        if split=='train':
            last_index = 0
            for samp_per_subj, uniq_subj_name in zip(samples_per_subject, unique_subjects_names):
                amount_train_samples_subj = int(floor(samp_per_subj * 0.8))
                train_subj_with_paths = []
                for i in range(last_index, len(subjects_with_pc_paths)):
                    if subjects_with_pc_paths[i][0] == uniq_subj_name:
                        train_subj_with_paths.append(subjects_with_pc_paths[i])
                        if len(train_subj_with_paths) == amount_train_samples_subj:
                            last_index = i
                            break
                assert len(train_subj_with_paths) == amount_train_samples_subj
                self.datapath += train_subj_with_paths
                # for tswp in train_subj_with_paths:
                #     print('train_subj_with_paths:', tswp)
                # for dp in self.datapath:
                #     print('self.datapath:', dp)
                # raw_input('PAUSED...')
                # print('------------------')
                
        elif split=='test':
            last_index = 0
            for samp_per_subj, uniq_subj_name in zip(samples_per_subject, unique_subjects_names):
                amount_train_samples_subj = int(floor(samp_per_subj * 0.8))
                amount_test_samples_subj = samp_per_subj - amount_train_samples_subj
                test_subj_with_paths = []
                for i in range(last_index+amount_train_samples_subj, len(subjects_with_pc_paths)):
                    if subjects_with_pc_paths[i][0] == uniq_subj_name:
                        test_subj_with_paths.append(subjects_with_pc_paths[i])
                        if len(test_subj_with_paths) == amount_test_samples_subj:
                            last_index = i+1
                            break
                assert len(test_subj_with_paths) == amount_test_samples_subj
                self.datapath += test_subj_with_paths
                # for tswp in test_subj_with_paths:
                #     print('test_subj_with_paths:', tswp)
                # for dp in self.datapath:
                #     print('self.datapath:', dp)
                # raw_input('PAUSED...')
                # print('------------------')
        
        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (point_set, cls) tuple

        if shuffle is None:
            if split == 'train': self.shuffle = True
            else: self.shuffle = False
        else:
            self.shuffle = shuffle

        self.reset()


    def _augment_batch_data(self, batch_data):
        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
    
        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data
        return provider.shuffle_points(rotated_data)


    def _readply(self, file):
        with open(file, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            vertices = np.zeros(shape=(num_verts, 3), dtype=np.float32)
            vertices[:,0] = plydata['vertex'].data['x']
            vertices[:,1] = plydata['vertex'].data['y']
            vertices[:,2] = plydata['vertex'].data['z']
            # vertices[:,3] = plydata['vertex'].data['red']
            # vertices[:,4] = plydata['vertex'].data['green']
            # vertices[:,5] = plydata['vertex'].data['blue']
            # print('plydata:', plydata)
            # print('vertices:', vertices)
            # sys.exit(0)
            return vertices


    def _get_item(self, index): 
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)

            # Bernardo
            print('lfw_3Dreconstructed_MICA_dataset.py: _get_item(): loading file:', fn[1])

            # point_set = np.loadtxt(fn[1],delimiter=',').astype(np.float32)   # original
            if fn[1].endswith('.npy'):
                point_set = np.load(fn[1]).astype(np.float32)                    # Bernardo
            elif fn[1].endswith('.ply'):
                point_set = self._readply(fn[1]).astype(np.float32)            # Bernardo

            # Bernardo
            if point_set.shape[1] == 7:        # if contains curvature
                point_set = point_set[:,:-1]   # remove curvature column

            # Take the first npoints
            point_set = point_set[0:self.npoints,:]
            if self.normalize:
                point_set[:,0:3] = pc_normalize(point_set[:,0:3])
            if not self.normal_channel:
                point_set = point_set[:,0:3]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
        return point_set, cls
        
    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.datapath)

    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 3

    def reset(self):
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath)+self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.datapath))
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.npoints, self.num_channel()))
        batch_label = np.zeros((bsize), dtype=np.int32)
        for i in range(bsize):
            ps,cls = self._get_item(self.idxs[i+start_idx])
            batch_data[i] = ps
            batch_label[i] = cls
        self.batch_idx += 1
        if augment: batch_data = self._augment_batch_data(batch_data)
        return batch_data, batch_label
    
if __name__ == '__main__':
    d = ModelNetDataset(root = '../data/modelnet40_normal_resampled', split='test')
    print(d.shuffle)
    print(len(d))
    import time
    tic = time.time()
    for i in range(10):
        ps, cls = d[i]
    print(time.time() - tic)
    print(ps.shape, type(ps), cls)

    print(d.has_next_batch())
    ps_batch, cls_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(cls_batch.shape)
