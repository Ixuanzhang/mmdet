from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import range
import matplotlib.pyplot as plt
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)

try:
  FONT = ImageFont.truetype('arial.ttf', 24)   # 指定字号和大小
except IOError:
  FONT = ImageFont.load_default()

def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, font, color='black', thickness=1):
  draw = ImageDraw.Draw(image)
  (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  text_bottom = ymin
  # Reverse list and print from bottom to top.
  text_width, text_height = font.getsize(display_str)
  margin = np.ceil(0.05 * text_height)

  draw.rectangle(
      [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                        text_bottom)],
      fill=color)
  draw.text(
      (left + margin, text_bottom - text_height - margin),
      display_str,
      fill='black',
      font=font)

  return image

def draw_bounding_boxes(image, gt_boxes,class_name):
  num_boxes = gt_boxes.shape[0]  # box的数目
  gt_boxes_new = gt_boxes.copy()
  disp_image = Image.fromarray(np.uint8(image))  # array转换成image

  for i in range(num_boxes):
    this_class = int(gt_boxes_new[i, 4])-1
    disp_image = _draw_single_box(disp_image,
                                int(gt_boxes_new[i, 0]),
                                int(gt_boxes_new[i, 1]),
                                int(gt_boxes_new[i, 0])+int(gt_boxes_new[i,2]),
                                int(gt_boxes_new[i, 1])+int(gt_boxes_new[i,3]),
                                class_name[this_class],
                                FONT,
                                color=STANDARD_COLORS[this_class % NUM_COLORS])

  # image[0, :] = np.array(disp_image)
  image = np.array(disp_image)
  return image

if __name__ == '__main__':
    # class name
    class_name = ['excavator']
    save_path = '/home/dl/zx/Do/pytorch/mmdetection-master/outs/testVisual'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # img_path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/test2017/qingxie_DJI_0813.JPG'
    # img = Image.open(img_path)
    # a_new_img = img.resize((1200,800))
    # a_new_img.save('a.jpg')
    img_path = '/home/dl/zx/Do/pytorch/mmdetection-master/dataDeal/a.jpg'
    aa_img = Image.open(img_path)
    disp_image = Image.fromarray(np.uint8(aa_img))
    # print(new_img.type)
    x_max = 715.5880126953125+34.5125732421875
    y_max = 318.7242736816406+44.7691650390625
    gt_boxes = [715.5880126953125, 318.7242736816406, x_max, y_max ,1]
    boxes = np.array(gt_boxes)
    this_class = int(boxes[4]) - 1
    new_img = _draw_single_box(disp_image,
                     int(boxes[0]),
                     int(boxes[1]),
                     int(boxes[2]),
                     int(boxes[3]),
                     class_name[this_class],
                     FONT,
                     color=STANDARD_COLORS[this_class % NUM_COLORS])
    b_image = np.array(new_img)
    image = Image.fromarray(b_image)
    img_save_path = save_path + '/'  + 'b.jpg'
    image.save(img_save_path)



# if __name__ == '__main__':
#     # class name
#     class_name = ['excavator']
#     save_path = '/home/dl/zx/Do/pytorch/mmdetection-master/outs/testVisual'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     imgs_path = '/home/zx/zxfile/lab/Do/pytorch/YOLOv3-model-pruning-master/data/VisDrone2019/images/val'
#     labels_path = '/home/zx/zxfile/lab/Do/pytorch/YOLOv3-model-pruning-master/data/VisDrone2019/annotations/val'
#     label_files = os.listdir(labels_path)
#     i=0
#     for label_file in label_files:
#         img_name = label_file[:-4]
#         img_path = imgs_path + '/' +img_name + '.jpg'
#         img = Image.open(img_path)
#         with open(labels_path + '/'+label_file,'r') as file:
#             lines = file.readlines()
#             gt_boxes = []
#             for line in lines:
#                 line = line.strip('\n').split(',')
#                 if line[4] !='0':
#                     gt_boxes.append(line)
#             boxes = np.array(gt_boxes)
#             new_img = draw_bounding_boxes(img,boxes,class_name)
#             image = Image.fromarray(new_img)
#             img_save_path = save_path + '/' + img_name + '.jpg'
#             image.save(img_save_path)
#         i+=1
#         print(img_name + 'is dealed!')
#     print("%d images is dealed!" % i)