#coding = utf-8

import os
# path = '/home/dl/zx/Do/tensorflow/tensorflow-yolov3-master/data/images/rotate'
#train_path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/ann/test'
# test_path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/ann/test'
train_path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/no_car_img'
img_path_name = ['qingxie','shigongdiansucai','shoudong','zhengshe']



for item in img_path_name:
    images_list = os.listdir(os.path.join(train_path,item))
    total_num = len(images_list)
    num = 0
    for image in images_list:
        if image.endswith('.JPG'):
            src = os.path.join(os.path.abspath(train_path),item,image)
            dst = os.path.join(os.path.abspath(train_path),item,item+'_'+image)   # 这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式

            try:
                os.rename(src,dst)
                num+=1
                #print('convert %s to %s..' % (src,dst))
            except:
                continue

    print (item + '  total %d to rename & converted %d jpgs' % (total_num, num))
