#coding = utf-8
import os
import numpy as np
import shutil

data_path = '/home/dl/zx/guanxian/dataset'
save_imgs_path_train = '/home/dl/zx/guanxian/train_val_test/train/data'
save_xmls_path_train = '/home/dl/zx/guanxian/train_val_test/train/ann'
save_imgs_path_test = '/home/dl/zx/guanxian/train_val_test/test/data'
save_xmls_path_test = '/home/dl/zx/guanxian/train_val_test/test/ann'
imgs_path_name = ['qingxie','shigongdiansucai','shoudong','zhengshe']
xmls_path_name = ['qingxie_xml','shigongdiansucai_xml','shoudong_xml','zhengshe_xml']

for i,item in enumerate(xmls_path_name):
    print(item)
    xml_path = os.path.join(data_path, item)
    img_path = os.path.join(data_path,imgs_path_name[i])
    save_img_path_train = os.path.join(save_imgs_path_train, imgs_path_name[i])
    save_xml_path_train = os.path.join(save_xmls_path_train, imgs_path_name[i])
    save_img_path_test = os.path.join(save_imgs_path_test, imgs_path_name[i])
    save_xml_path_test = os.path.join(save_xmls_path_test, imgs_path_name[i])

    xml_list = os.listdir(xml_path)
    np.random.seed(100)
    np.random.shuffle(xml_list)
    train_ratio = 0.85
    train_num = int(len(xml_list) * train_ratio)
    # print(train_num)
    xml_list_train = xml_list[:train_num]
    xml_list_val = xml_list[train_num:]
    dealed_train_num=0
    for xml_item in xml_list_train:
        img_name = xml_item.split('.')[0] + '.JPG'
        shutil.copyfile(os.path.join(img_path,img_name),os.path.join(save_img_path_train,img_name))
        shutil.copyfile(os.path.join(xml_path,xml_item),os.path.join(save_xml_path_train,xml_item))
        dealed_train_num+=1
    for xml_item in xml_list_val:
        img_name = xml_item.split('.')[0] + '.JPG'
        shutil.copyfile(os.path.join(img_path,img_name),os.path.join(save_img_path_test,img_name))
        shutil.copyfile(os.path.join(xml_path,xml_item),os.path.join(save_xml_path_test,xml_item))

    print(item + '  train num is %d,  has been dealed num is %d' %(train_num,dealed_train_num))
