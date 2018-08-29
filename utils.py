import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import cv2
import math
from config import cfg
import scipy.misc as misc

# classes
class_list = ['Car', 'Van' , 'Truck' , 'Pedestrian' , 'Person_sitting' , 'Cyclist' , 'Tram' ]

bc={}
bc['minX'] = 0; bc['maxX'] = 80; bc['minY'] = -40; bc['maxY'] = 40
bc['minZ'] =-2; bc['maxZ'] = 1.25


def removePoints(PointCloud, BoundaryCond):
    
    # Boundary condition
    minX = BoundaryCond['minX'] ; maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY'] ; maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ'] ; maxZ = BoundaryCond['maxZ']
    
    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0]<=maxX) & (PointCloud[:, 1] >= minY) & (PointCloud[:, 1]<=maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2]<=maxZ))
    PointCloud = PointCloud[mask]
    return PointCloud

def makeBVFeature(PointCloud_, BoundaryCond, Discretization):
    # 1024 x 1024 x 3
    Height = 1024+1
    Width = 1024+1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:,0] = np.int_(np.floor(PointCloud[:,0] / Discretization))
    PointCloud[:,1] = np.int_(np.floor(PointCloud[:,1] / Discretization) + Width/2)
    
    # sort-3times
    indices = np.lexsort((-PointCloud[:,2],PointCloud[:,1],PointCloud[:,0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height,Width))

    _, indices = np.unique(PointCloud[:,0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]
    #some important problem is image coordinate is (y,x), not (x,y)
    heightMap[np.int_(PointCloud_frac[:,0]), np.int_(PointCloud_frac[:,1])] = PointCloud_frac[:,2]


    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height,Width))
    densityMap = np.zeros((Height,Width))
    
    _, indices, counts = np.unique(PointCloud[:,0:2], axis = 0, return_index=True,return_counts = True)
    PointCloud_top = PointCloud[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1)/np.log(64))
    
    intensityMap[np.int_(PointCloud_top[:,0]), np.int_(PointCloud_top[:,1])] = PointCloud_top[:,3]
    densityMap[np.int_(PointCloud_top[:,0]), np.int_(PointCloud_top[:,1])] = normalizedCounts
    """
    plt.imshow(densityMap[:,:])
    plt.pause(2)
    plt.close()
    plt.show()
    plt.pause(2)
    plt.close()
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    plt.imshow(intensityMap[:,:])
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    """
    RGB_Map = np.zeros((Height,Width,3))
    RGB_Map[:,:,0] = densityMap      # r_map
    RGB_Map[:,:,1] = heightMap       # g_map
    RGB_Map[:,:,2] = intensityMap    # b_map
    
    save = np.zeros((512,1024,3))
    save = RGB_Map[0:512,0:1024,:]
    #misc.imsave('test_bv.png',save[::-1,::-1,:])
    #misc.imsave('test_bv.png',save)   
    return save


def generate_rgbmap(pointcloud):
    '''
    kitti_loader
    x [0, 40]
    y [-40, 40]
    z [-2, 1.25]

    1024 * 512
    n * m
    rslidar_loader
    [-40, 40]
    [-2, 2]
    输入点云
    返回 n * m * 3
    '''
    # 筛选范围内的点
    lidar_coord = np.array([-cfg.X_MIN, -cfg.Y_MIN, -cfg.Z_MIN], dtype=np.float32)
    voxel_size = np.array([cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE], dtype=np.float32)
    # print(voxel_size)
    # min(1.0, log(N+1)/64)
    r_map = np.zeros([cfg.RGB_Map_M, cfg.RGB_Map_N], dtype=np.float32)
    # max z
    g_map = np.zeros([cfg.RGB_Map_M, cfg.RGB_Map_N], dtype=np.float32)
    # max i
    b_map = np.zeros([cfg.RGB_Map_M, cfg.RGB_Map_N], dtype=np.float32)

    bound_x = np.logical_and(
        pointcloud[:, 0] >= cfg.X_MIN, pointcloud[:, 0] < cfg.X_MAX)
    bound_y = np.logical_and(
        pointcloud[:, 1] >= cfg.Y_MIN, pointcloud[:, 1] < cfg.Y_MAX)
    bound_z = np.logical_and(
        pointcloud[:, 2] >= cfg.Z_MIN, pointcloud[:, 2] < cfg.Z_MAX)

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    pointcloud = pointcloud[bound_box]

    shifted_coord = pointcloud[:, :3] + lidar_coord
    voxel_index = np.floor(shifted_coord[:, :2] / voxel_size).astype(np.int)   # 体素index

    for idx in range(pointcloud.shape[0]):

        z = pointcloud[idx][2]
        i = pointcloud[idx][3]

        index_x, index_y = voxel_index[idx]
        r_map[index_x, index_y] += 1

        if g_map[index_x, index_y] < z:
            g_map[index_x, index_y] = z

        if b_map[index_x, index_y] < i:
            b_map[index_x, index_y] = i

    r_map = np.log(r_map + 1) / 64.0
    r_map = np.minimum(r_map, 1.0)

    r_map = r_map[..., np.newaxis]
    g_map = g_map[..., np.newaxis]
    b_map = b_map[..., np.newaxis]

    rgb_map = np.concatenate((r_map, g_map, b_map), axis=2)

    return rgb_map

def get_target(label_file, Tr, R0):
    """
    :param label_file: label文件
    :param Tr: 标定参数
    :param R0: 标定参数
    :return: 将camera坐标系转换成雷达坐标系
    """
    target = np.zeros([50, 7], dtype=np.float32)
    with open(label_file, 'r') as f:
        lines = f.readlines()
    num_obj = len(lines)
    index = 0
    for j in range(num_obj):
        obj = lines[j].strip().split(' ')
        obj_class = obj[0].strip()
        if obj_class in class_list:
             t_lidar , box3d_corner = box3d_cam_to_velo(obj[8:], Tr, R0)   # get target  3D object location h,w,l, x,y,z,ry

             location_x = t_lidar[0][0]          
             location_y = t_lidar[0][1]

             if(location_x > 0) & (location_x < 40) & (location_y > -40) & (location_y < 40):

                  target[index][2] = t_lidar[0][0]/40              # make sure target inside the covering area (0,1)
                  target[index][1] = (t_lidar[0][1]+40)/80         # we should put this in [0,1] ,so divide max_size  80 m
                  obj_width = obj[9].strip()
                  obj_length = obj[10].strip()
                  target[index][3] = float(obj_width)/80
                  target[index][4] = float(obj_length)/40          # get target width ,length
                  obj_alpha = obj[14].strip()                       # get target Observation angle of object, ranging [-pi..pi]
                  target[index][5] = math.sin(float(obj_alpha))    # complex YOLO   Im
                  target[index][6] = math.cos(float(obj_alpha))    # complex YOLO   Re
                  #print(np.arctan2(target[0][4],target[0][5]))
                  for i in range(len(class_list)):
                       if obj_class == class_list[i]:
                           target[index][0] = i
                  index = index+1
    """
    p0, y, x, w, l, im, re    y,x,w,l 都归一化
    """
    return target

def box3d_cam_to_velo(box3d, Tr, R0):
    """
    :param box3d: camera 坐标系 w,h,l,x,y,z,ry
    :param Tr: 标定参数
    :param R0: 标定参数
    :return: lidar坐标系下的坐标
    """
    def project_cam2velo(cam, Tr, R0):
        """
        :param cam: (4, 1)   tx, ty, tz, 1
        :param Tr: (4, 4)
        :param R0: (4, 4)
        :return: (3, 1)
        """
        R_cam_to_rect = np.eye(4)
        R_cam_to_rect[:3, :3] = np.array(R0).reshape(3, 3)
        cam = np.matmul(np.linalg.inv(R_cam_to_rect), cam)
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)
    def ry_to_rz(ry):
        angle = -ry - np.pi / 2
        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2*np.pi + angle
        return angle

    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr, R0)
    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])
    rz = ry_to_rz(ry)
    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])
    velo_box = np.dot(rotMat, Box)                            # (3, 8)
    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T
    box3d_corner = cornerPosInVelo.transpose()                # (8, 3)
    return t_lidar, box3d_corner.astype(np.float32)

def load_kitti_calib(calib_file):

    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

anchors = [1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52]

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
        Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
        my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
        My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]
    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

if __name__ == '__main__':

    c1 = torch.FloatTensor([2, 2, 4, 4])
    c2 = torch.FloatTensor([1, 1, 4, 4])

    a = bbox_iou(c1, c2, x1y1x2y2=False)
    print(a)

    #
    # lidar_file = '/home/yxk/data/Kitti/object/training/velodyne/000025.bin'
    # a = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    # b = removePoints(a, bc)
    # data = makeBVFeature(b, 40.0 / 512)
    # # print(data.shape)
    # # misc.imsave('test_bv5.png', data[:, :, :])
    #
    # rgb = generate_rgbmap(a)
    # print(rgb.shape)
    # misc.imsave('test_bv6.png', rgb[:, :, :])
    # ww = np.zeros([512, 1024, 3])
    #
    # ww[:, :, 0] = ww[:, :, 0] + rgb[:, :, 0]
    # misc.imsave('test_bv7.png', ww)
    #
    # ww = np.zeros([512, 1024, 3])
    # ww[:, :, 1] = ww[:, :, 1] + rgb[:, :, 1]
    # misc.imsave('test_bv8.png', ww)
    #
    # ww = np.zeros([512, 1024, 3])
    # ww[:, :, 2] = ww[:, :, 2] + rgb[:, :, 2]
    # misc.imsave('test_bv9.png', ww)

