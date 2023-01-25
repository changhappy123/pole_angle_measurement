# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime 
# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#from samples.coco import coco

'''
1.修改 权重文件.h5
2. 类别数量
3. 修改类别名称
'''

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_WEIGHT = 'mask_rcnn_shapes_0040.h5'
MODEL_WEIGHT = os.path.join(MODEL_DIR, MODEL_WEIGHT)

"""
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")
"""
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

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
    NUM_CLASSES = 1 + 1  # background + 3 shapes   #修改类别数量

    #修改的参数：
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    #BACKBONE = "resnet50"  # 主干网络

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    #VALIDATION_STEPS = 50



#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MODEL_WEIGHT, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'pole']   #修改类别名称
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

for file_name in file_names:
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
    a=datetime.now()
    # Run detection
    results = model.detect([image], verbose=1)
    b=datetime.now()
    # Visualize results
    print("time:",(b-a).seconds)
    r = results[0]
    print (r)
    #添加保存掩码图片
    # if r['rois'].shape[0]:
    #     for i in range(r['masks'].shape[-1]):
    #         mask_img=r['masks'][:,:,i].astype(int)*255
    #         mask_img=np.uint8(mask_img)
    #         cv2.imwrite('./result/mask_'+datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'.png',mask_img)

    # if r['rois'].shape[0]:
    #     mask_img=r['masks'].astype(int)*255
    #     mask_img=np.uint8(mask_img)
    #     cv2.imwrite('./result/mask_'+datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'.png',mask_img)


    '''
    #添加直线检测
    all_mask=r['mask']  
    angles=[]
    for i in range(all_mask.shape[-1]): #遍历每个掩码目标获取一个角度
        angle=detect_angle(all_mask[:,:,i])
        angles.append(angle)
    '''
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])


