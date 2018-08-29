import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from torch.nn import DataParallel

from complexYOLO import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss
# from loss2 import RegionLoss
from tensorboardX import SummaryWriter
from eval import draw_image_train
from config import cfg
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

writer = SummaryWriter()

batch_size = 8

data_dir = cfg.DATA_DIR

# dataset
dataset = KittiDataset(root=data_dir, set='train')
data_loader = data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=False)

# val data
val_set = KittiDataset(root=data_dir, set='val')
val_loader = data.DataLoader(val_set, batch_size, shuffle=True, pin_memory=False)


# model = DataParallel(ComplexYOLO())
# model = ComplexYOLO()
model = torch.load('ComplexYOLO_epoch%s' % 130)
model = DataParallel(model)
model.cuda()

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.0005)

# define loss function
region_loss = RegionLoss(num_classes=8, num_anchors=5)


for epoch in range(200):
   for group in optimizer.param_groups:
       if(epoch>=4 & epoch<80):
           group['lr'] = 1e-4
       if(epoch>=80 & epoch<160):
           group['lr'] = 1e-5
       if(epoch>=160):
           group['lr'] = 1e-6


    # 加载数据进行训练
   for batch_idx, (rgb_map, target) in enumerate(data_loader):

       optimizer.zero_grad()

       rgb_map = rgb_map.view(rgb_map.data.size(0), rgb_map.data.size(3), rgb_map.data.size(1), rgb_map.data.size(2))
       output = model(rgb_map.float().cuda())
       loss, loss_x, loss_y, loss_w, loss_l, loss_im, loss_re, loss_Euler, loss_conf, loss_cls = region_loss(output, target)

       loss.backward()
       optimizer.step()

       # 标量可视化
       if (batch_idx) % 30 == 0:
           writer.add_scalar("loss", loss, batch_idx + (310 * (epoch)))
           writer.add_scalar("loss_x", loss_x, batch_idx + (310 * (epoch)))
           writer.add_scalar("loss_y", loss_y, batch_idx + (310 * (epoch)))
           writer.add_scalar("loss_w", loss_w, batch_idx + (310 * (epoch)))
           writer.add_scalar("loss_l", loss_l, batch_idx + (310 * (epoch)))
           writer.add_scalar("loss_im", loss_im, batch_idx + (310 * (epoch)))
           writer.add_scalar("loss_re", loss_re, batch_idx + (310 * (epoch)))
           writer.add_scalar("loss_Euler", loss_Euler, batch_idx + (310 * (epoch)))
           writer.add_scalar("loss_conf", loss_conf, batch_idx + (310 * (epoch)))
           writer.add_scalar("loss_cls", loss_cls, batch_idx + (310 * (epoch)))

   print('train epoch [%d/%d], iter[%d/%d], lr %.5f, aver_loss %.5f' % (epoch, 200, batch_idx, len(data_loader), group['lr'], loss))

   # 模型保存
   if (epoch % 10 == 0):
       torch.save(model, "./model_save/ComplexYOLO_epoch"+str(epoch))
       draw_image_train(epoch)
