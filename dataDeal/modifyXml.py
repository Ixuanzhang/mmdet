# modify the filename and path in  xml  file
# implement conversion from xml to txt
# https://www.cnblogs.com/rainsoul/p/6283231.html

#coding:utf-8

import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import sys

path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/ann/val'
file_list = os.listdir(path)    # read all file dir in a path
new_name = 'excavator'
i=0

for xml_file in file_list:  #
    i+=1

    xml_file = xml_file.strip('\n')  # delete the line break
    xml_file_path = os.path.join(path, xml_file)
    tree = ET.parse(xml_file_path)  # open the file of xml depends on path (not only file name)
    root = tree.getroot()  # get the root

    for anyObject in root.findall('object'):
        name = anyObject.find('name').text  # 子节点下节点name的值
        if name=='forklift' or name == 'truck':
            anyObject.find('name').text = new_name
            print(xml_file +' '+ name + ' is converted to '+new_name)
    tree.write(xml_file_path,xml_declaration=True)
print('%d xml is converted！' %i)
