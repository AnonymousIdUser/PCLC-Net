import argparse
import os
import numpy as np
import pandas as pd
import cv2
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils_.config import cfg_from_yaml_file
from utils_ import misc
from datasets.io import IO
from datasets.data_transforms import Compose
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import json


def generate_random_Rt() -> (np.ndarray,  np.ndarray,  np.ndarray):
    rot = R.random()
    displacement = np.random.randn(3) # the x/y/z displacement from original point
    displacement[0:2] *= 20
    displacement[2] *= 5
    
    return rot.as_matrix(), rot.as_quat(), displacement

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_config', 
        help = 'yaml config file')
    parser.add_argument(
        'model_checkpoint', 
        help = 'pretrained weight')
    parser.add_argument('--pc_root', type=str, default='', help='Pc root')
    parser.add_argument('--pc', type=str, default='', help='Pc file')  
    parser.add_argument('--pose_gt_path', type=str, default='/data/xhm/dataset/PCN_Pose/test/pose_gt/03001627.csv', help='pose gt file')  
    parser.add_argument('--pose_mode', type=str, default='random_pose', help='pose mode in [only_rotate|random_pose|none]') 
    parser.add_argument('--category', type=str, default='table', help='the category to be compeletion') 
    parser.add_argument('--classification_file', type=str, default='/data/xhm/PointNeXt/log/scansalon/scansalon-finetune-pointnext-s-ngpus1-seed3819-20240117-221550-N9t8GPm2UPVvkroihsWP6S/scansalon_test_predict_lables.json', help='classification result of 10 class pc') 
    parser.add_argument('--dataset', type=str, default='pcn', help='the dataset name') 
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='whether to save img of complete point cloud') 
    parser.add_argument(
        '--out_pc_root',
        type=str,
        default='',
        help='root of the output pc file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config is not None
    assert args.model_checkpoint is not None
    assert (args.pc != '') or (args.pc_root != '')

    return args

def inference_single(model, pc_path, args, config, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path
    # read single point cloud
    pc_ndarray = IO.get(pc_file).astype(np.float32)
    # r, _, _ = generate_random_Rt() 
    # pc_ndarray = np.dot(r, pc_ndarray.T).T
    # read the gt pose
    if args.pose_mode == 'only_rotate':
        displacement = np.mean(pc_ndarray, axis=0)
        pc_ndarray = pc_ndarray - displacement
    # transform it according to the model 

    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    
    pc_ndarray_normalized = transform({'input': pc_ndarray})
    # inference
    ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
    dense_points = ret[-6] 
    # coarse_points = ret[-7]
    shift_dis = ret[-3]
    pred_R = ret[-5]
    pred_T = ret[-4]
    r_pred = pred_R
    t_pred = pred_T + shift_dis.squeeze()
    dense_points = model.transform_pc(dense_points, r_pred, t_pred).squeeze(0).detach().cpu().numpy()

    if args.pose_mode == 'only_rotate':
        pc_ndarray = pc_ndarray + displacement
        dense_points = dense_points + displacement
    if args.out_pc_root != '':
        target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
        os.makedirs(target_path, exist_ok=True)

        # np.save(os.path.join(target_path, 'fine.npy'), dense_points)
        input_pc = o3d.geometry.PointCloud()
        output_pc = o3d.geometry.PointCloud()
        if args.category == 'none' and args.dataset == 'pcn':
            gt_path = root.replace('partial', 'complete')
            if gt_path[-1] == '/':
                gt_path = gt_path[:-1]
            gt_path = gt_path + '.pcd'
        else:
            gt_path = os.path.join(root.replace('partial', 'complete'), pc_path)
        if os.path.exists(gt_path) == True:
            gt_pc = o3d.io.read_point_cloud(gt_path)
        input_pc.points = o3d.utility.Vector3dVector(pc_ndarray)
        output_pc.points = o3d.utility.Vector3dVector(dense_points)
        o3d.io.write_point_cloud(os.path.join(target_path, "input.ply"), input_pc, write_ascii=True)
        o3d.io.write_point_cloud(os.path.join(target_path,  "output.ply"), output_pc, write_ascii=True)
        if os.path.exists(gt_path) == True:
            o3d.io.write_point_cloud(os.path.join(target_path,  "gt.ply"), gt_pc, write_ascii=True)
        if args.save_vis_img:
            input_img = misc.get_ptcloud_img(pc_ndarray_normalized['input'].numpy())
            dense_img = misc.get_ptcloud_img(dense_points)
            cv2.imwrite(os.path.join(target_path, 'input.png'), input_img)
            cv2.imwrite(os.path.join(target_path, 'fine.png'), dense_img)
    
    return

def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()
    classes = ["trash_bin", "cabinet", "chair", "car", "lamp", "bookshelf", "table", "bed", "sofa", "bathhub"]
    category_id = {"chair":"03001627", "sofa":"04256520", "car":"02958343", "lamp":"03636649", "table":"04379243", 
                   "bookshelf":"02871439", "cabinet":"02933112", "bathhub":"02808440", "trash_bin":"02747177", "bed":"02818832"}

    if args.pc_root != '':
        pc_file_list = os.listdir(args.pc_root)
        files = pc_file_list
        for pc_file in pc_file_list:
            if pc_file in files and (pc_file[-3:] == 'ply' or pc_file[-3:] == 'pcd' or pc_file[-3:] == 'obj'):
                inference_single(base_model, pc_file, args, config, root=args.pc_root)
    else:
        inference_single(base_model, args.pc, args, config)

if __name__ == '__main__':
    main()