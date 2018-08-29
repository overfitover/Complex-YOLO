import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
from torch.optim import Adam, SGD
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import kitti
import kitti_aug


import torch.utils.data as data
from complexYOLO import ComplexYOLO
import torch.optim as optim
from region_loss import RegionLoss
from eval import draw_image
from config import cfg

# argumentparse
parser = ArgumentParser()
parser.add_argument('-bs', '--batch_size', type=int, default=12, help="batch size of the data")
parser.add_argument('-e', '--epochs', type=int, default=4000, help='epoch of the train')
parser.add_argument('-c', '--n_class', type=int, default=8, help='the classes of the dataset')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

# import visualize
writer = SummaryWriter()

batch_size = args.batch_size
learning_rate = args.learning_rate
epoch_num = args.epochs
n_class = args.n_class
use_cuda = torch.cuda.is_available()
data_dir = cfg.DATA_DIR

print('load data....')
dataset=kitti.KittiDataset(root=data_dir, set='train')
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=5)

val_data=kitti.KittiDataset(root=data_dir, set='val')
val_loader = torch.utils.data.DataLoader(val_data,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=5)

model = ComplexYOLO()
if use_cuda:
    model.cuda()
# define optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.0005)
# define loss function
region_loss = RegionLoss(num_classes=8, num_anchors=5)
def train(epoch):
    model.train()
    for batch_idx, (rgb_map, target) in enumerate(train_loader):
        optimizer.zero_grad()
        rgb_map = rgb_map.view(rgb_map.data.size(0), rgb_map.data.size(3), rgb_map.data.size(1), rgb_map.data.size(2))
        output = model(rgb_map.float().cuda())
        loss, loss_x, loss_y, loss_w, loss_l, loss_im, loss_re, loss_Euler, loss_conf, loss_cls = region_loss(output, target)
        loss.backward()
        optimizer.step()
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
            # writer.add_image('Image', rgb_map, batch_idx)
        if (batch_idx) % 20 == 0:
            print('train epoch [%d/%d], iter[%d/%d], lr %.5f, aver_loss %.5f' % (epoch,
                                                                                 epoch_num, batch_idx,
                                                                                 len(train_loader), learning_rate,
                                                                                 loss))
    # model save
    if (epoch % 10 == 0):
        torch.save(model, "./model_save/ComplexYOLO_epoch" + str(epoch))
        draw_image(epoch)


def val(epoch):
    model.eval()
    for batch_idx, (rgb_map, target) in enumerate(train_loader):
        rgb_map = rgb_map.view(rgb_map.data.size(0), rgb_map.data.size(3), rgb_map.data.size(1), rgb_map.data.size(2))
        output = model(rgb_map.float().cuda())
        loss, loss_x, loss_y, loss_w, loss_l, loss_im, loss_re, loss_Euler, loss_conf, loss_cls = region_loss(output, target)
        if (batch_idx + 1) % 1 == 0:
            print('test epoch [%d/%d], iter[%d/%d], aver_loss %.5f' % (epoch, epoch_num, batch_idx, len(val_loader), loss))

        writer.add_scalar("val_loss", loss, (epoch))
        writer.add_scalar("val_loss_x", loss_x, (epoch))
        writer.add_scalar("val_loss_y", loss_y, (epoch))
        writer.add_scalar("val_loss_w", loss_w, (epoch))
        writer.add_scalar("val_loss_l", loss_l, (epoch))
        writer.add_scalar("val_loss_im", loss_im, (epoch))
        writer.add_scalar("val_loss_re", loss_re, (epoch))
        writer.add_scalar("val_loss_Euler", loss_Euler, (epoch))
        writer.add_scalar("val_loss_conf", loss_conf, (epoch))
        writer.add_scalar("val_loss_cls", loss_cls, (epoch))
        break

if __name__ == '__main__':
    for epoch in range(epoch_num):
        train(epoch)
        val(epoch)
        # adjust learning rate
        if epoch == 40:
            learning_rate *= 0.1
            optimizer.param_groups[0]['lr'] = learning_rate
            # optimizer.param_groups[1]['lr'] = learning_rate * 2
        elif epoch == 60:
            learning_rate *= 0.1
            optimizer.param_groups[0]['lr'] = learning_rate