#!/usr/bin/env bash

GPUS=$1
test_dir=$2

CUDA_VISIBLE_DEVICES=${GPUS} 

# 使用时请注释下面某一个(针对所用数据集注释另一个数据集对应脚本)
# 更换测试样例种类时需要切换下面脚本中对应类别模型权重的文件地址

######################################## Inference on PCN ########################################  

for file in ${test_dir}/*;
do
    echo ${file##*/}
    python tools/inference.py cfgs/PCNPose_models/PCLCNet.yaml \
            ./experiments/PCLCNet/PCNPose_models/PCNPose_models_chair/ckpt-best.pth  \
            --pc_root=${file} \
            --out_pc_root=inference_result/PCNPose_models_chair/${file##*/}/\
            --pose_mode=none \
            --category=none
done

######################################## Inference on ScanSalon ########################################  
python tools/inference.py cfgs/ScanSalon_models/PCLCNet.yaml \
        ./experiments/PCLCNet/ScanSalon_models/Scansalon_models_chair/ckpt-best.pth  \
        --pc_root=${test_dir} \
        --out_pc_root=inference_result/ScanSalon_models/03001627/\
        --pose_mode=none \
        --category=none \
        --dataset=scansalon

# PCN测试样例(椅子类)
# bash ./scripts/inference.sh 0 ./data/PCN/test/partial/03001627
# ScanSalon测试样例(椅子类)
# bash ./scripts/inference.sh 0 ./data/ScanSalon/test/partial/03001627