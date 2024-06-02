import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils_.logger import *
import torch
import pandas as pd
import open3d as o3d
from sklearn import cluster

def point_cloud_kmeans_cluster(pcd_path, cluster_num=8):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.array(pcd.points)
    
    kmeans = cluster.KMeans(n_clusters=cluster_num, random_state=42, n_init=10, init="k-means++")
    kmeans.fit(points)  
    labels = kmeans.labels_ # get labels of points
    
    return labels

category_id = {"chair":"03001627", "sofa":"04256520", "car":"02958343", "lamp":"03636649", "table":"04379243", 
                   "bookshelf":"02871439", "cabinet":"02933112", "bathhub":"02808440", "trash_bin":"02747177", "bed":"02818832"}


@DATASETS.register_module()
class ScanSalon(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.point_transform = config.TRANSFORM
        self.category = config.CATEGORY
        self.category_id = category_id[self.category]
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cd_file_path = config.CD_FILE_PATH % (self.subset, self.category_id)
        self.only_rotate = config.ONLYROTATE
        self.is_registration = config.REGISTRATION
        self.cd2_threshold = config.CDL2_THRESHOLD
        self.is_cd_select = config.CD_SELECT
        self.train_samples_num = config.TRAIN_SAMPLES_NUM
        # self.data_augmentation = config.DATA_AUGMENTATION

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if self.category != "all":
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] == self.category]
                # self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] in ['chair', 'sofa', 'car', 'lamp', 'table', 'cabinet', 'bathhub', 'bookshelf', 'bed', 'trash_bin']] #'desk'
        if self.is_cd_select == True and os.path.exists(self.cd_file_path):
            self.input_cd_dict = {}
            for dc in self.dataset_categories:
                input_cd = pd.read_csv(config.CD_FILE_PATH % (self.subset, dc['taxonomy_id']), sep=' ')
                self.input_cd_dict[dc['taxonomy_id']] = input_cd

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            if  self.point_transform == 'None':
                return data_transforms.Compose([{
                    'callback': 'KMeansClusterDrop',
                    'objects': ['partial']
                },{
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': 2048
                    },
                    'objects': ['partial']
                },{
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }])
            else:
                return data_transforms.Compose([{
                    'callback': 'KMeansClusterDrop',
                    'objects': ['partial']
                },{
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': 2048
                    },
                    'objects': ['partial']
                },{
                    'callback': self.point_transform,
                    'objects': ['partial', 'gt']
                },{
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }])
        else:
            if self.point_transform in ['FixPoseRotateAndDisplacementPoints', 'RandomRotateAndDisplacementPoints']:
                return data_transforms.Compose([{
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': 2048
                    },
                    'objects': ['partial']
                }, {
                    'callback': self.point_transform,
                    'objects': ['partial', 'gt']
                }, {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }])
            else:
                return data_transforms.Compose([{
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': 2048
                    },
                    'objects': ['partial']
                }, {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []
        # if self.data_augmentation == True:
        #     self.points_kmeans_labels = {}

        if self.is_registration:
            for dc in self.dataset_categories:
                print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='ScanSalonDATASET')
                samples = dc[subset]
                if self.subset == "train": # for few-shot learning
                    samples = samples[0 : self.train_samples_num]

                for s in samples:
                    if self.is_cd_select == True:
                        # if self.subset == "train" and self.input_cd[self.input_cd.id == s]['CD2'].values[0] >= self.cd2_threshold:
                        #     continue
                        if subset != "train" and self.input_cd_dict[dc['taxonomy_id']][self.input_cd_dict[dc['taxonomy_id']].id == s]['CD2'].values[0] >= 0.1:
                            continue
                    file_list.append({
                        'taxonomy_id':
                        dc['taxonomy_id'],
                        'model_id':
                        s,
                        'partial_path': 
                        self.partial_points_path % (subset, dc['taxonomy_id'], s),
                        'gt_path': 
                        self.complete_points_path % (subset, dc['taxonomy_id'], s),
                    })
                    # if self.data_augmentation == True:
        else:
            for dc in self.dataset_categories:
                print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='ScanSalonDATASET')
                samples = dc[subset]
                if self.subset == "train": # for few-shot learning
                    samples = samples[0 : self.train_samples_num]

                for s in samples:
                    if self.is_cd_select == True:
                        if self.subset != "train" and self.input_cd_dict[dc['taxonomy_id']][self.input_cd_dict[dc['taxonomy_id']].id == s]['CD2'].values[0] >= 0.1:
                            continue
                    file_list.append({
                        'taxonomy_id':
                        dc['taxonomy_id'],
                        'model_id':
                        s,
                        'partial_path': 
                        self.partial_points_path % (subset, dc['taxonomy_id'], s),
                        # [
                        #     self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        #     for i in range(n_renderings)
                        # ],
                        'gt_path':
                        self.complete_points_path % (subset, dc['taxonomy_id'], s),
                    })
        # if self.is_registration:
        #     print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % ("03001627", "chair"), logger='ScanSalonDATASET')
        #     samples = os.listdir("../dataset/ScanSalon_Equi_Pose/partial/%s/" % ("chair"))

        #     for s in samples:
        #         file_list.append({
        #             'taxonomy_id':
        #             "03001627",
        #             'model_id':
        #             s,
        #             'partial_path': 
        #             self.partial_points_path % ("chair", s),
        #             'gt_path': 
        #             self.complete_points_path % ("chair", s),
        #         })
        # else:
        #     print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % ("03001627", "chair"), logger='ScanSalonDATASET')
        #     samples = os.listdir("../dataset/ScanSalon_Equi_Pose/partial/%s/" % ("chair"))

        #     for s in samples:
        #         file_list.append({
        #             'taxonomy_id':
        #             "03001627",
        #             'model_id':
        #             s,
        #             'partial_path': 
        #             self.partial_points_path % ("chair", s),
        #             'gt_path':
        #             self.complete_points_path % ("chair", s),
        #             })
        

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='ScanSalonDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        # rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            # print(file_path)
            # if type(file_path) == list:
            #     file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)
        if self.subset != "train" and self.only_rotate:
            gt_displacement = torch.mean(data['gt'], dim=0)
            partial_displace_ment = torch.mean(data['partial'], dim=0)
            if self.subset == "test":
                data['partial'] = data['partial'] - partial_displace_ment
            elif self.subset == "val":
                data['partial'] = data['partial'] - gt_displacement
            data['gt'] = data['gt'] - gt_displacement
        # center_pt = (data['partial'].max(dim = 0, keepdim=True)[0] + data['partial'].min(dim = 0, keepdim=True)[0]) / 2
        center_pt = data['partial'].mean(dim=0, keepdim=True)
        if self.subset != "train" and self.point_transform == "RandomRotateAndDisplacementPoints":
            # print(center_pt, data['partial'].mean(dim=0, keepdim=True))
            data['partial'] -= center_pt
            data['gt'] -= center_pt
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'], center_pt)

    def __len__(self):
        return len(self.file_list)