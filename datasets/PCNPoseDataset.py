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


@DATASETS.register_module()
class PCNPose(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.point_transform = config.TRANSFORM
        self.category = config.CATEGORY
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS
        self.only_rotate = config.ONLYROTATE
        self.is_registration = config.REGISTRATION

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']
            if self.category != "all":
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] == self.category]
                # self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] in ['chair', 'sofa', 'car', 'table', 'lamp', 'bookshelf', 'cabinet', 'bed', 'bathhub', 'trash_bin']]

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            if  self.point_transform == 'None':
                return data_transforms.Compose([{
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': 2048
                    },
                    'objects': ['partial']
                }, {
                    'callback': 'RandomMirrorPoints',
                    'objects': ['partial', 'gt']
                },{
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
                    'callback': 'RandomMirrorPoints',
                    'objects': ['partial', 'gt']
                },{
                    'callback': self.point_transform,
                    'objects': ['partial', 'gt']
                },{
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }])
        else:
            if self.point_transform in ['FixPoseRotateAndDisplacementPoints', 'RandomRotateAndDisplacementPoints']:
            # if self.point_transform in ['FixPoseRotateAndDisplacementPoints']:
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

        if self.is_registration:
            for dc in self.dataset_categories:
                print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
                samples = dc[subset]

                for s in samples:
                    file_list.append({
                        'taxonomy_id':
                        dc['taxonomy_id'],
                        'model_id':
                        s,
                        'partial_path': [
                            self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                            for i in range(n_renderings)
                        ],
                        'gt_path': [
                            self.complete_points_path % (subset, dc['taxonomy_id'], s + f"_0{i}")
                            for i in range(n_renderings)
                        ]
                        
                    })
        else:
            for dc in self.dataset_categories:
                print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
                samples = dc[subset]

                for s in samples:
                    file_list.append({
                        'taxonomy_id':
                        dc['taxonomy_id'],
                        'model_id':
                        s,
                        'partial_path': [
                            self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                            for i in range(n_renderings)
                        ],
                        'gt_path':
                        self.complete_points_path % (subset, dc['taxonomy_id'], s),
                    })
        

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
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
        # if self.subset != "train" and self.point_transform == "RandomRotateAndDisplacementPoints":
        #     # print(center_pt, data['partial'].mean(dim=0, keepdim=True))
        #     data['partial'] -= center_pt
        #     data['gt'] -= center_pt
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'], center_pt)

    def __len__(self):
        return len(self.file_list)