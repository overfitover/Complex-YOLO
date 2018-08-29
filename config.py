#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
相关参数配置
'''
import os
import os.path as osp
from easydict import EasyDict as edict

__C = edict()
# import config as cfg
cfg = __C

# for data set type
# __C.DATA_SETS_TYPE = 'kitti'
# __C.DATA_SETS_TYPE = 'rslidar16'
__C.DATA_SETS_TYPE = 'kitti'
# 输入数据
if __C.DATA_SETS_TYPE == 'kitti':
    # __C.DATA_DIR = '/media/disk4/deeplearningoflidar/maxwell/data'
    #__C.DATA_DIR = '/mnt/sdc4/1inux/data/kitti/for_voxelnet/cropped_dataset/'
    __C.DATA_DIR = '/home/yxk/data/Kitti/object'
    #__C.DATA_DIR = '/media/disk1/maxwell/data/kitti/for_voxelnet/cropped_dataset/'
    #__C.DATA_DIR = '/home/yxk/project/data/'

elif __C.DATA_SETS_TYPE == 'rslidar16':
    __C.DATA_DIR = '/media/DataCenter/label/16/total'

__C.CALIB_DIR = '/mnt/1inux/data/kitti/data_object_calib/training/calib'

if __C.DATA_SETS_TYPE == 'kitti':
    __C.IMAGE_WIDTH = 1242
    __C.IMAGE_HEIGHT = 375
    __C.IMAGE_CHANNEL = 3

# for gpu allocation
__C.GPU_AVAILABLE = '0'
__C.GPU_USE_COUNT = len(__C.GPU_AVAILABLE.split(','))
__C.GPU_MEMORY_FRACTION = 1

# 网络参数

__C.X_MIN = 0.0
__C.X_MAX = 40.0
__C.Y_MIN = -40.0
__C.Y_MAX = 40.0
__C.Z_MIN = -2.0
__C.Z_MAX = 1.25

__C.RGB_Map_N = 1024
__C.RGB_Map_M = 512
__C.FEATURE_NUM = 3

__C.CELL_SIZE_X = 16
__C.CELL_SIZE_Y = 32

__C.VOXEL_X = (__C.X_MAX - __C.X_MIN)
__C.VOXEL_Y = (__C.Y_MAX - __C.Y_MIN)

__C.VOXEL_X_SIZE = (__C.X_MAX - __C.X_MIN) / __C.RGB_Map_M
__C.VOXEL_Y_SIZE = (__C.Y_MAX - __C.Y_MIN) / __C.RGB_Map_N

# print('__C.VOXEL_X_SIZE', __C.VOXEL_X_SIZE)
# print('__C.VOXEL_Y_SIZE', __C.VOXEL_Y_SIZE)

__C.ALPHA = 0.1

__C.OBJECT_SCALE = 5
__C.NOOBJECT_SCALE = 1
__C.CLASS_SCALE = 1
__C.COORD_SCALE = 1
__C.COORD_EULER_SCALE = 1

__C.NUM_CLASSES = 8
__C.NUM_ANCHORS = 5
# E-RPN Grid
# 32 * 16

# anchor box 3
# vehicle size heading up heading down
# 0 180
__C.ANCHOR_VEHICLE_L = 3.9
__C.ANCHOR_VEHICLE_W = 1.6

# 0 180
__C.ANCHOR_CYCLIST_L = 1.76
__C.ANCHOR_CYCLIST_W = 0.6
# -90
__C.ANCHOR_PEDESTRIAN_L = 0.8
__C.ANCHOR_PEDESTRIAN_W = 0.6

# total anchor
__C.ANCHORS = [[3.9, 1.6], [3.9, 1.6],
              [1.76, 0.6], [1.76, 0.6],
              [0.8, 0.6]]

__C.LABELNAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
'''
Pedestrian : (1.48, 1.97, 1.75389912280702)
Car : (1.22, 2.22, 1.522310740354517)
Cyclist : (1.44, 1.9, 1.748611422172449)
Person_sitting : (1.06, 1.64, 1.2734939759036144)
Tram : (2.99, 3.84, 3.5285714285714307)
Truck : (2.4, 4.2, 3.1627392739273743)
Misc : (0.76, 3.91, 2.061084905660379)
Van : (1.65, 2.86, 2.2222881880024707)
'''
__C.HEIGHT_PREDEFINED = [1.52, 2.22, 3.16, 1.75, 1.27, 1.75, 3.53, 2.06]

# cyclist size heading up down

# pedestrian size(heading left)

__C.BV_LOG_FACTOR = 1 
__C.INPUT_HEIGHT = 1024
__C.INPUT_WIDTH = 512

lambda_coord = 1

__C.MATRIX_P2 = ([[719.787081,    0.,            608.463003, 44.9538775],
                    [0.,            719.787081,    174.545111, 0.1066855],
                    [0.,            0.,            1.,         3.0106472e-03],
                    [0.,            0.,            0.,         0]])

# cal mean from train set
__C.MATRIX_T_VELO_2_CAM = ([
    [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
    [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
    [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
    [0, 0, 0, 1]
])

# 去除变换

# __C.MATRIX_T_VELO_2_CAM = ([
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [1, 0, 0, 0],
#     [0, 0, 0, 1]
# ])

# cal mean from train set

__C.MATRIX_R_RECT_0 = ([
    [0.99992475, 0.00975976, -0.00734152, 0],
    [-0.0097913, 0.99994262, -0.00430371, 0],
    [0.00729911, 0.0043753, 0.99996319, 0],
    [0, 0, 0, 1]
])

# __C.MATRIX_R_RECT_0 = ([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ])

# for rpn nms
__C.RPN_NMS_POST_TOPK = 20
__C.RPN_NMS_THRESH = 0.1
__C.RPN_SCORE_THRESH = 0.96

__C.CORNER2CENTER_AVG = True