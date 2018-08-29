# Complex-YOLO
Complex-YOLO: Real-time 3D Object Detection on Point Clouds   pytorch

# Introduction
This is an unofficial implementation of Complex-YOLO: Real-time 3D Object Detection on Point Clouds in pytorch. A large part of this project is based on the work here:https://github.com/marvis/pytorch-yolo2


Point Cloud Preprocessing is based on:https://github.com/skyhehe123/VoxelNet-pytorch


# Data Preparation

Download the 3D KITTI detection dataset.

Camera calibration matrices of object data set (16 MB)

Training labels of object data set (5 MB)

Velodyne point clouds (29 GB)


# Train

python3 main.py
or
python train.py
train.py 对比了测试集和训练集在训练过程中的loss变化情况, 为了观察是否过拟合.


# usage
在config.py 文件中修改文件所在的路径

todo

1.训练的结果在训练集和测试集上表现的差异很大. 现在不确定原因.
2.我想增加显示中间层特征的方案



