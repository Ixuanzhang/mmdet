import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread


# filepath = '/home/zx/zxfile/contest/deep-person-reid-master/torchreid/data/contestdata/query_a/'  # 数据集目录
filepath = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/train2017'
pathDir = os.listdir(filepath)

R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    #print(pathDir[idx])
    if pathDir[idx][-4:] == '.JPG':
        print(idx)
        filename = pathDir[idx]
        img = imread(os.path.join(filepath, filename)) / 255.0
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])

num = len(pathDir) * 5472 * 3648  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    if pathDir[idx][-4:] == '.JPG':
        print(idx)
        trainlename = pathDir[idx]
        img = imread(os.path.join(filepath, filename)) / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

R_var = np.sqrt(R_channel / num)
G_var = np.sqrt(G_channel / num)
B_var = np.sqrt(B_channel / num)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))

'''  train
R_mean is 0.568774, G_mean is 0.641165, B_mean is 0.549302
R_var is 0.240394, G_var is 0.222515, B_var is 0.220638
'''

'''  test
R_mean is 0.101572, G_mean is 0.180105, B_mean is 0.210071
R_var is 0.252135, G_var is 0.163426, B_var is 0.203572
'''

