import os
from PIL import Image
# img_file:图片的路径
# path_save:保存路径
# width：宽度
# height：长度

def img_resize(img_file, path_save, width,height):
    img = Image.open(img_file)
    new_image = img.resize((width,height),Image.BILINEAR)
    new_image.save(path_save)

if __name__ =='__main__':
    # imgs_path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/test2017_ori'
    # last_save_path = '/home/dl/zx/Do/pytorch/mmdetection-master/data/coco/test2017'
    imgs_path = '/home/dl/zx/ori_image'
    last_save_path = '/home/dl/zx/image'
    resize_width = 1200
    resize_height = 800
    img_list = os.listdir(imgs_path)
    for img in img_list:
        img_path = os.path.join(imgs_path,img)
        save_path = os.path.join(last_save_path,img)
        img_resize(img_path,save_path,resize_width,resize_height)