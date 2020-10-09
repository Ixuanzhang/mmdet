# implement conversion from xml to txt
# https://www.cnblogs.com/rainsoul/p/6283231.html



#coding:utf-8

import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import sys

label_convert_id = {'excavator': 0, 'forklift': 1, 'truck': 2}
path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/VOCdevkit/VOC2007/train_val_test/train/ann'
images_path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/VOCdevkit/VOC2007/JPEGImages/train'
name = ['qingxie','shigongdiansucai','shoudong','zhengshe']

# i=0   #  the number of dealing with xml file
train_dataset_file = open('our_train.txt','w')
#test_dataset_file = open('our_test.txt','w')

for item in name:
    xml_path = os.path.join(path,item)
    img_path = os.path.join(images_path,item)
    file_list = os.listdir(xml_path)  # read all file dir in a path
    for xml_file in file_list:  #
        # need filename
        #      xmin ,ymin,xmax,ymax
        xml_file = xml_file.strip('\n')   # delete the line break
        xml_file_path = os.path.join(xml_path,xml_file)
        tree = ET.parse(xml_file_path)    # open the file of xml depends on path (not only file name)
        root = tree.getroot()        # get the root
        # print("*"*10)
        filename = root.find('filename').text
        filename = filename[:-4]
        print(filename)
        # i=i+1

        bndbox_list = []
        # 找到root节点下的所有object节点  so use root.findall()
        for anyObject in root.findall('object'):
            name = anyObject.find('name').text   # 子节点下节点name的值
            # print(name)
            bndbox = anyObject.find('bndbox')    # 子节点下属性bndbox的值
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            bndbox_val = [xmin, ymin, xmax, ymax, label_convert_id[name]]
            bndbox_list.append(bndbox_val)     # save all box in one image
            #print(bndbox_val)
        # print(bndbox_list)

        ''' save to classes.txt '''
        image_path = os.path.join(images_path,item,filename+'.JPG')
        train_dataset_file.write(image_path)
        #
        for i in range(len(bndbox_list)):
            train_dataset_file.write(' ' + str(bndbox_list[i][0])+','+str(bndbox_list[i][1])+','+str(bndbox_list[i][2])+','+str(bndbox_list[i][3])+','+str(bndbox_list[i][4]))
        train_dataset_file.write('\n')

#print(i)