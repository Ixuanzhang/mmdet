import os
import json
import numpy as np
import glob
import shutil
import xml.etree.ElementTree as ET
np.random.seed(41)
# 从所有的图片中选出没车的图片（即：选出没对应xml文件的图片）

#0为背景
classname_to_id = {'excavator': 1, 'forklift': 2, 'truck': 3}
def get(root, name):
    return root.findall(name)
def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

class Lableimg2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    def to_coco(self, xml_path_list,img_path):
        self._init_categories()

        for index,xml_file in enumerate(xml_path_list):
            xml_f = xml_file
            tree = ET.parse(xml_f)
            root = tree.getroot()

            #构建COCO的image字段
            image = {}
            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)
            img_name = xml_file.split('/')[-1].split('.')[0]
            image['height'] = height
            image['width'] = width
            image['id'] = self.img_id
            image['file_name'] = os.path.join(img_name+'.JPG')
            self.images.append(image)

            # 构建COCO的annotation字段
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                category_id = classname_to_id[category]
                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
                ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
                xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
                ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                    self.img_id, 'bbox': [xmin, ymin, o_width, o_height],
                       'category_id': category_id, 'id': self.ann_id,
                       'segmentation': []}
                self.annotations.append(ann)
                self.ann_id += 1

            self.img_id += 1


        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance


    ## Cruuently we do not support segmentation
    #  segmented = get_and_check(root, 'segmented', 1).text
    #  assert segmented == '0'

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)




if __name__ == '__main__':
    train_labelimg_path = "/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/ann/train"
    test_labelimg_path = "/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/ann/test"
    train_img_path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/train2017'
    val_img_path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/val2017'
    saved_coco_path = "./"
    # 创建文件
    if not os.path.exists("%scoco/annotations/"%saved_coco_path):
        os.makedirs("%scoco/annotations/"%saved_coco_path)

    train_xml_list = glob.glob(train_labelimg_path + "/*.xml")
    test_xml_list = glob.glob(test_labelimg_path + "/*.xml")
    print("train_n:", len(train_xml_list), 'val_n:', len(test_xml_list))

    # 把训练集转化为COCO的json格式
    l2c_train = Lableimg2CoCo()
    train_instance = l2c_train.to_coco(train_xml_list,train_img_path)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json'%saved_coco_path)

    # 把验证集转化为COCO的json格式
    l2c_val = Lableimg2CoCo()
    val_instance = l2c_val.to_coco(test_xml_list,val_img_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json'%saved_coco_path)
