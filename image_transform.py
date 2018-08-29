import scipy.misc as misc
import numpy as np


image_array = np.random.randint(0, 255, size=[1024, 512, 3])
misc.imsave('2.jpg', image_array)
#
# misc.toimage(image_array, cmin=0, cmax=255).save('2.jpg')
#
#
# from PIL import Image
#
# a = np.array(Image.open('3.jpg'))
# b = [255, 255, 255] - a
# im = Image.fromarray(b.astype('uint8'))
# im.save('4.jpg')
#
# image = Image.fromarray(image_array.astype('uint8'))
# image.save('5.jpg')
#
#
# import cv2
#
# cv2.imwrite("6.png", image_array)



# from PIL import Image
# import numpy as np
#
# a = np.asarray(Image.open('3.jpg').convert('L')).astype('float')
# depth = 10.
# grad = np.gradient(a)   # 取图像灰度的梯度值
# grad_x, grad_y = grad   # 分别取横纵图像梯度值
# grad_x = grad_x*depth/100.
# grad_y = grad_y*depth/100.
# A = np.sqrt(grad_x**2+grad_y**2+1.)
# uni_x = grad_x/A
# uni_y = grad_y/A
# uni_z = 1./A
#
# vec_el = np.pi/2.2  # 光源的俯视角度, 弧度制
# vec_az = np.pi/4.   # 光源的方位角度, 弧度制
# dx = np.cos(vec_el)*np.cos(vec_az)  # 光源对X轴的影响
# dy = np.sin(vec_el)*np.sin(vec_az)  # 光源对Y轴的影响
# dz = np.sin(vec_el)  # 光源对Z轴的影响
#
# b = 255*(dx * uni_x + uni_y * dy + dz*uni_z)  # 光源归一化
# b = b.clip(0, 255)
# im = Image.fromarray(b.astype('uint8'))
# im.save('33.jpg')



