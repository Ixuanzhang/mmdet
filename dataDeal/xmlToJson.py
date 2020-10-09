# coding:utf-8
# convert xml to generate json
# pip install lxml
# 将测试得到的图片中的框转换为json文件

import os
import glob
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET

path2 = "."

START_BOUNDING_BOX_ID = 1


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


if __name__ == '__main__':
    # classes = ['excavator', 'forklift', 'truck']
    classes = ['excavator', 'forklift', 'truck']
    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i
    save_json = 'data.json'
    xmls_path_name = ['qingxie_xml','shigongdiansucai_xml','shoudong_xml','zhengshe_xml']
    all_xml_path = '/home/dl/zx/guanxian/dataset'
    # json_dict = {"folder": [], "images": [], "annotations": []}
    json_dict = {"images":[]}
    i=0
    for xml_path_name in xmls_path_name:
        xml_path = os.path.join(all_xml_path,xml_path_name)
        xml_list = glob.glob(xml_path + "/*.xml")
        xml_list = np.sort(xml_list)
        categories = pre_define_categories.copy()
        folder_name = xml_path_name.split('_')[0]
        print(folder_name)
        for index, line in enumerate(xml_list):
            xml_f = line
            tree = ET.parse(xml_f)
            root = tree.getroot()

            filename = os.path.basename(xml_f)[:-4] + ".jpg"
            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)

            one_img_ann = []
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                category_id = pre_define_categories[category]

                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
                ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
                xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
                ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
                assert (xmax > xmin), "xmax <= xmin, {}".format(line)
                assert (ymax > ymin), "ymax <= ymin, {}".format(line)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {'bbox': [xmin, ymin, o_width, o_height],
                       'category_id': category_id}
                one_img_ann.append(ann)
            image = {'file_name': filename, 'folder': folder_name, 'height': height, 'width': width,'annotations':one_img_ann}
            json_dict['images'].append(image)
            i+=1

    with open(save_json,'w') as f:
        json.dump(json_dict,f)
    print('%d xml file is converted!' % i)
