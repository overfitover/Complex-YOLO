"""
@description:这是测试输出中间层特征得例子
"""
import torch
from kitti import KittiDataset
import torch.utils.data as data
import scipy.misc as misc

batch_size = 12
model = torch.load('ComplexYOLO_epoch%s' % 130)
model.cuda()
print(model)

# dataset
dataset = KittiDataset(root='/home/yxk/data/Kitti/object', set='train')
data_loader = data.DataLoader(dataset, batch_size, num_workers=6, shuffle=True, pin_memory=False)

for batch_idx, (rgb_map, target) in enumerate(data_loader):
    rgb_map = rgb_map.view(rgb_map.data.size(0), rgb_map.data.size(3), rgb_map.data.size(1), rgb_map.data.size(2))
    x = rgb_map.float().cuda()


    for name, module in model._modules.items():
        x = module(x)
        print(x.shape)

        model = torch.load('ComplexYOLO_epoch%s' % 70)
        # if name == 'bn_14':
        #     print(x.shape)
        #     a = x[0, :1, :, :]
        #     print(a.shape)
        #
        #     # misc.imsave('eval_bv' + str(1) + '.png', x[0, :, :, 0])
        #
        #     break


