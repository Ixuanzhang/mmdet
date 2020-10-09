# coding = utf-8

import os
import shutil
import glob

data_path = '/home/dl/zx/guanxian/dataset'
save_img_path = '/home/dl/zx/guanxian/no_car_img'
xmls_path_name = ['qingxie_xml','shigongdiansucai_xml','shoudong_xml','zhengshe_xml']
for item in xmls_path_name:
    xmls_path = os.path.join(data_path,item)
    imgs_path = os.path.join(data_path,item.split('_')[0])
    imgs_list = os.listdir(imgs_path)
    xmls_list = os.listdir(xmls_path)
    num=0
    for img in imgs_list:
        if img[-4:]=='.JPG':
            to_xml_name = img.split('.')[0]+'.xml'
            if to_xml_name not in xmls_list:
                shutil.copyfile(os.path.join(imgs_path,img),os.path.join(save_img_path,item.split('_')[0],img))
                num+=1
    print(item+'  %d image no has car!' %num )