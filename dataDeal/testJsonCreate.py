import os
import json
import numpy as np
import glob
import shutil
import xml.etree.ElementTree as ET
np.random.seed(41)

#0为背景
# classname_to_id = {'excavator': 1, 'forklift': 2, 'truck': 3}   # 多类
classname_to_id = {'excavator': 1}  # 单类

if __name__ == '__main__':
    test_img_path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/test2017'
    saved_coco_path = "./"
    # 创建文件
    if not os.path.exists("%scoco/annotations/"%saved_coco_path):
        os.makedirs("%scoco/annotations/"%saved_coco_path)

    categories = []
    for k, v in classname_to_id.items():
        category = {}
        category['id'] = v
        category['name'] = k
        categories.append(category)

    # 构建COCO的image字段
    images=[]
    annotations = []
    img_list = os.listdir(test_img_path)
    img_id = 1
    ann_id = 1
    i=0
    for img in img_list:
        i+=1
        print(i)
        image = {}
        width = 5472
        height =3648
        image['height'] = height
        image['width'] = width
        image['id'] = img_id
        image['file_name'] = img
        images.append(image)

        # 构建COCO的annotation字段
        ann = {'area': 0, 'iscrowd': 0, 'image_id':
            img_id, 'bbox': [],
               'category_id': 0, 'id': ann_id,
               'segmentation': []}
        annotations.append(ann)
        ann_id += 1
        img_id += 1
    instance = {}
    instance['info'] = 'spytensor created'
    instance['license'] = ['license']
    instance['images'] = images
    instance['annotations'] = annotations
    instance['categories'] = categories

    save_path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/annotations/instances_test2017.json'
    json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示