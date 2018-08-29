import numpy as np
import cv2
import os
import multiprocessing as mp
import argparse
import glob
from tools import *


object_dir = '/home/yxk/data/Kitti/object/training'
def aug_data(tag, object_dir=object_dir, name='0'):

    np.random.seed()
    lidar = np.fromfile(os.path.join(object_dir,
                                     'velodyne', tag + '.bin'), dtype=np.float32).reshape(-1, 4)
    label = np.array([line for line in open(os.path.join(
        object_dir, 'label_2', tag + '.txt'), 'r').readlines()])  # (N')
    cls = np.array([line.split()[0] for line in label])  # (N')

    gt_box3d = label_to_gt_box3d(np.array(label)[np.newaxis, :], cls='', coordinate='camera')[0]  # (N', 7) x, y, z, h, w, l, r

    choice = np.random.randint(0, 10)
    if choice >= 7:
        # disable this augmention here. current implementation will decrease the performances
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_corner_gt_box3d = center_to_corner_box3d(
            lidar_center_gt_box3d, coordinate='lidar')
        for idx in range(len(lidar_corner_gt_box3d)):
            # TODO: precisely gather the point
            is_collision = True
            _count = 0
            while is_collision and _count < 100:
                t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
                t_x = np.random.normal()
                t_y = np.random.normal()
                t_z = np.random.normal()
                # check collision
                tmp = box_transform(
                    lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar')
                is_collision = False
                for idy in range(idx):
                    x1, y1, w1, l1, r1 = tmp[0][[0, 1, 4, 5, 6]]
                    x2, y2, w2, l2, r2 = lidar_center_gt_box3d[idy][[
                        0, 1, 4, 5, 6]]
                    iou = cal_iou2d(np.array([x1, y1, w1, l1, r1], dtype=np.float32),
                                    np.array([x2, y2, w2, l2, r2], dtype=np.float32))
                    if iou > 0:
                        is_collision = True
                        _count += 1
                        break
            if not is_collision:
                box_corner = lidar_corner_gt_box3d[idx]
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(
                    lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(
                    lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(
                    lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(
                    np.logical_and(bound_x, bound_y), bound_z)
                lidar[bound_box, 0:3] = point_transform(
                    lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
                lidar_center_gt_box3d[idx] = box_transform(
                    lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar')

        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = name+'aug_{}_1_{}'.format(tag, np.random.randint(1, 1024))

    elif choice < 7 and choice >= 4:
        # global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_center_gt_box3d = box_transform(lidar_center_gt_box3d, 0, 0, 0, r=angle, coordinate='lidar')
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = name+'aug_{}_2_{:.4f}'.format(tag, angle).replace('.', '_')

    else:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_center_gt_box3d[:, 0:6] = lidar_center_gt_box3d[:, 0:6] * factor
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = name+'aug_{}_3_{:.4f}'.format(tag, factor).replace('.', '_')

    label = box3d_to_label(gt_box3d[np.newaxis, ...], cls[np.newaxis, ...], coordinate='camera')[0]  # (N')
    return newtag, lidar, label


def worker(tag, name='0'):
    new_tag, lidar, label = aug_data(tag, name=name)
    output_path = os.path.join(object_dir, 'training_aug')


    lidar.reshape(-1).tofile(os.path.join(output_path, 'velodyne', new_tag + '.bin'))

    with open(os.path.join(output_path, 'label_2', new_tag + '.txt'), 'w+') as f:
        for line in label:
            f.write(line)

    with open(os.path.join(output_path, 'train.txt'), 'a') as f:
        f.write(new_tag)
        f.write('\n')
    # print(new_tag)



if __name__ == '__main__':

    with open(os.path.join(object_dir, 'train.txt')) as f:
        file_list = f.read().splitlines()

    for tag in file_list:
        worker(tag, name='0')

