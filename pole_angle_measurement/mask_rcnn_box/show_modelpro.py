# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
#import utils
from mrcnn import model as modellib,utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image
"""
加入自己类别名称
更改类别个数
更改 label
"""

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Root directory of the project
ROOT_DIR = os.getcwd()

#ROOT_DIR = os.path.abspath("../")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num=0

# Local path to trained weights file  #权重文件
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes    #修改类别数量

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


config = ShapesConfig()
config.display()

class DrugDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image,image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.  加载类别信息和图片信息
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        #self.add_class("shapes", 1, "tank") # 黑色素瘤   修改类别名称
        # self.add_class("shapes", 1, "category1")
        # self.add_class("shapes", 2, "category2")
        self.add_class("shapes", 1, "pole") #添加类别信息

        for i in range(count): #编历每张图片
            # 获取图片宽和高

            filestr = imglist[i].split(".")[0] #图片名
            
            mask_path = mask_floder + "/" + filestr + ".png" #掩码
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml" #info.yaml文件
            #自己添加json路径
            #json_path= dataset_root_path + "json/"+ filestr +".json"  #json文件
            print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png") #rgb

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)
            # self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
            #                width=cv_img.shape[1], height=cv_img.shape[0],
            #                mask_path=mask_path, yaml_path=yaml_path,json_path=json_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id",image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img,image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            # if labels[i].find("category1") != -1:
            #     labels_form.append("category1")
            # if labels[i].find("category2") != -1:
            #     labels_form.append("category2")
            if labels[i].find("pole") != -1:   #修改label
                labels_form.append("pole")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

#基础设置
dataset_root_path="mydata/label/"
img_floder = dataset_root_path + "pic"
mask_floder = dataset_root_path + "cv2_mask"
imglist = os.listdir(img_floder)
count = len(imglist)

#train与val数据集准备
dataset = DrugDataset()  #加载所有训练图片
dataset.load_shapes(count, img_floder, mask_floder, imglist,dataset_root_path)
dataset.prepare()

print("dataset-->",dataset._image_ids)

from mrcnn import utils
from mrcnn import model
imgid=list(range(30))
augmentation=None
augment=False

import cv2
def showimg(img):
    cv2.imshow('result',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#原始box
def extract_bboxes_2(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1

            # #添加 边框扩宽，增加特征识别
            # nx2 =  x2 + 2*(x2-x1) if x2 + 2*(x2-x1)< mask.shape[1] else mask.shape[1]
            # nx1 = x1 - 2 * (x2 - x1) if x1 - 2 * (x2 - x1) > 0 else 0
            # ny1 = y1 - (x2-x1) if  y1 - (x2-x1)>0 else 0
            # ny2 = y2
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
            #nx1, nx2, ny1, ny2 = 0, 0, 0, 0

        boxes[i] = np.array([y1, x1, y2, x2])  #修改
        #boxes[i] = np.array([ny1, nx1, ny2, nx2])

    return boxes.astype(np.int32)

'''
#path='mydata/label/pic/171206_062637090_Camera_6.jpg'
image_id=35

image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
    model.load_image_gt(dataset, config, image_id, augment=augment,
                  augmentation=augmentation,
                  use_mini_mask=config.USE_MINI_MASK)

#showimg(image)
#cv2.imwrite('./001.jpg',image)
#box=[[164 241 241 251], [190 240 241 245]]
for i in range(len(gt_boxes)):
    cv2.rectangle(image,tuple(gt_boxes[i][[1,0]]),tuple(gt_boxes[i][[3,2]]),(0,0,255),2)
cv2.resize(image,(800,600))
showimg(image)
'''

##############展示边框和掩码
# Load random image and mask.
image_ids = np.random.choice(dataset.image_ids,10)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)
    box = extract_bboxes_2(mask)

    # Display image and additional stats
    print("image_id ", image_id, dataset.image_reference(image_id))
    # log("image", image)
    # log("mask", mask)
    # log("class_ids", class_ids)
    # log("bbox", bbox)
    for i in range(len(box)):
        y1,x1,y2,x2=box[i]
        py1,px1,py2,px2 = bbox[i]
        gt_angle= np.arctan((x2-x1)/(y2-y1))*180/np.pi
        pre_angle = np.arctan((px2 - px1) / (py2 - py1)) * 180 / np.pi
        print('imageID:',image_id)
        print('gt:',gt_angle)
        print('pre:', pre_angle)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names,box=box)


####################################
#展示数据集生成器和输入边框
''''
for image_id in imgid:
    image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
        model.load_image_gt(dataset, config, image_id, augment=augment,
                      augmentation=augmentation,
                      use_mini_mask=config.USE_MINI_MASK)

    #showimg(image)
    #cv2.imwrite('./001.jpg',image)
    #box=[[164 241 241 251], [190 240 241 245]]
    for i in range(len(gt_boxes)):
        cv2.rectangle(image,tuple(gt_boxes[i][[1,0]]),tuple(gt_boxes[i][[3,2]]),(0,0,255),2)
    cv2.resize(image,(800,600))
    showimg(image)
'''
####################################
'''
#解码过程
class_names = ['BG', 'pole']   #修改类别名称
score=np.array([1,1])

results = []
final_rois, final_class_ids, final_scores, final_masks = \
        unmold_detections(detections[i], mrcnn_mask[i],
                           image.shape, molded_images[i].shape,
                           windows[i])
results.append({
    "rois": final_rois,
    "class_ids": final_class_ids,
    "scores": final_scores,
    "masks": final_masks,
})

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
'''