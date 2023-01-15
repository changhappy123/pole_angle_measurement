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
import yaml
from mrcnn.model import log
from PIL import Image

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco

'''
1.修改 权重文件.h5
2. 类别数量
3. 修改类别名称
'''

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# MODEL_WEIGHT = './logs/shapes20191104T0033/mask_rcnn_shapes_0010.h5'
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

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 320
    # IMAGE_MAX_DIM = 384

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    #BACKBONE = "resnet50"  # 主干网络

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50




# import train_tongue
# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()   #环境配置

########################
# 2. 加载数据
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
    def draw_mask(self, num_obj, mask, image ,image_id):
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
        # self.add_class("shapes", 1, "tank") # 黑色素瘤   修改类别名称
        # self.add_class("shapes", 1, "category1")
        # self.add_class("shapes", 2, "category2")
        self.add_class("shapes", 1, "pole")  # 添加类别信息

        for i in range(count):  # 编历每张图片
            # 获取图片宽和高

            filestr = imglist[i].split(".")[0]  # 图片名

            mask_path = mask_floder + "/" + filestr + ".png"  # 掩码
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"  # info.yaml文件
            # 自己添加json路径
            # json_path= dataset_root_path + "json/"+ filestr +".json"  #json文件
            print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")  # rgb

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
        print("image_id" ,image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img ,image_id)
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
            if labels[i].find("pole") != -1:  # 修改label
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


# 基础设置
dataset_root_path ="mydata/label/"
img_floder = dataset_root_path + "pic"
mask_floder = dataset_root_path + "cv2_mask"
imglist = os.listdir(img_floder)
count = len(imglist)

# train与val数据集准备
dataset_train = DrugDataset()  # 加载所有训练图片
dataset_train.load_shapes(count, img_floder, mask_floder, imglist ,dataset_root_path)
dataset_train.prepare()

print("dataset_train-->" ,dataset_train._image_ids)

dataset = DrugDataset()
dataset.load_shapes(int(count *0.3), img_floder, mask_floder, imglist ,dataset_root_path)
dataset.prepare()

print("dataset_val-->" ,dataset._image_ids)

# 加载模型预测
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MODEL_WEIGHT, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'pole']  # 修改类别名称
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

'''
# 预测结果
image_id = random.choice(dataset.image_ids)
# 真实值
image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

info = dataset.image_info[image_id]
#image = skimage.io.imread(img_path)
a = datetime.now()
# Run detection
results = model.detect([image], verbose=1)
b = datetime.now()
# Visualize results
print("time:", (b - a).seconds)
r = results[0]
print(r)
# 添加保存掩码图片
if r['rois'].shape[0]:
    for i in range(r['masks'].shape[-1]):
        mask_img = r['masks'][:, :, i].astype(int) * 255
        mask_img = np.uint8(mask_img)
        #cv2.imwrite('./result/mask_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.png', mask_img)

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])  # 展示及保存预测结果

##################展示预测精度：
# Draw precision-recall curve  # AP ，R ，P
AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                     r['rois'], r['class_ids'], r['scores'], r['masks'])
visualize.plot_precision_recall(AP, precisions, recalls)


# Grid of ground truth objects and their predictions
visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
                        overlaps, dataset.class_names) # Compute VOC-style Average Precision

'''
##########计算平均  AP、P、R
# Compute VOC-style Average Precision # map
def compute_batch_ap(image_ids):
    APs = []
    APs75=[]
    P=[]
    R=[]
    P75= []
    R75 = []

    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], r['masks'])

        AP75, precisions75, recalls75, overlaps75 = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r['rois'], r['class_ids'], r['scores'], r['masks'],iou_threshold=0.75)

        recall, positive_ids,precision=utils.compute_recall(r['rois'], gt_bbox, iou=0.5)
        recall75, positive_ids75, precision75 = utils.compute_recall(r['rois'], gt_bbox, iou=0.75)

        APs75.append(AP75)
        APs.append(AP)
        P.append(precision)
        R.append(recall)
        P75.append(precision75)
        R75.append(recall75)

    return APs,P,R,APs75,P75,R75

# Pick a set of random images
image_ids = np.random.choice(dataset.image_ids, 50)
APs,P,R,APs75,P75,R75 = compute_batch_ap(image_ids)

print("mAP @ IoU=50: ", np.mean(APs))
print("mAP @ IoU=75: ", np.mean(APs75))
print("precisions: ", np.mean(P))
print("recalls: ", np.mean(R))
print("precisions75: ", np.mean(P75))
print("recalls75: ", np.mean(R75))

mAP=np.mean(APs)
mprecisions= np.mean(P)
mrecalls= np.mean(R)
visualize.plot_precision_recall(mAP, mprecisions, mrecalls)

















