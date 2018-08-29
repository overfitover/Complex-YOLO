from __future__ import division
import os
import os.path
import torch
import numpy as np
import cv2
import math

from utils import *

"""
从数据增强后数据集里加载数据 

data_aug数据增强保存起来后, 通过kitti_aug加载数据增强的数据
"""

class KittiDataset(torch.utils.data.Dataset):

    def __init__(self, root='/home/yxk/data/Kitti/object', set='train', type='velodyne_train'):
        self.type = type
        self.root = root
        self.data_path = os.path.join(root, 'training/training_aug')
        self.lidar_path = os.path.join(self.data_path, "velodyne/")
        # self.image_path = os.path.join(self.data_path, "image_2/")
        self.calib_path = os.path.join(self.root, 'training', "calib/")
        self.label_path = os.path.join(self.data_path, "label_2/")

        with open(os.path.join(self.data_path, '%s.txt' % set)) as f:
            self.file_list = f.read().splitlines()

    def __getitem__(self, i):

        calib_tag = self.file_list[i].split('_')[1]
        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.bin'
        calib_file = self.calib_path + '/' + calib_tag + '.txt'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'
        # image_file = self.image_path + '/' + self.file_list[i] + '.png'
        # print(self.file_list[i])

        if self.type == 'velodyne_train':

            calib = load_kitti_calib(calib_file)

            target = get_target(label_file, calib['Tr_velo2cam'], calib['R0'])
            # print(target)
            # print(self.file_list[i])

            ################################
            # load point cloud data
            a = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

            data = generate_rgbmap(a)

            # b = removePoints(a, bc)
            # data = makeBVFeature(b, bc, 40/512)   # (512, 1024, 3)

            return data, target

        elif self.type == 'velodyne_test':
            NotImplemented

        else:
            raise ValueError('the type invalid')

    def __len__(self):
        return len(self.file_list)


if __name__ == "__main__":
    a = KittiDataset(root='/home/yxk/data/Kitti/object', set='train')
    a.__getitem__(1)
