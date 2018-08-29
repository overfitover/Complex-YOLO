import numpy as np
from kitti.kitti_util import *
import cv2
import os

# with open("/home/yxk/data/Kitti/object/training/label_2/000001.txt") as f:
#     file_list = f.read().splitlines()
#
# print(file_list)
#
#
# for list in file_list:
#
#     label = Object3d(list)
#     label.print_object()


# cal = Calibration('/home/yxk/data/Kitti/object/training/calib/000000.txt')
#
# print(cal.b_x)
# print(cal.b_y)
# print(cal.V2C)
#

''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
'''

import os
import sys
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
import kitti.kitti_util as utils


root_dir = '/home/yxk/data/Kitti/object'

class kitti_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert (idx < self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        assert (idx < self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert (idx < self.num_samples and self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return utils.read_label(label_filename)

kitti = kitti_object(root_dir)
print(kitti.__len__())
print(kitti.get_label_objects(0))


