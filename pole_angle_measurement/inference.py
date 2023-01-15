import os
import sys
import skimage.io
from datetime import datetime
import numpy as np
import cv2

ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.join(ROOT_DIR,"mask_rcnn_box")
sys.path.append(ROOT_DIR)
from mask_rcnn_box.mrcnn.config import Config
import mask_rcnn_box.mrcnn.model as modellib
from mask_rcnn_box.mrcnn import visualize
from myutile.mask_to_pcd import  pointProcess,img_show
from myutile.cloud import  calcul_angle
from pole_pointnet import pointnet_inference

'''
1.修改 权重文件.h5
2. 类别数量
3. 修改类别名称
'''

#配置文件
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


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()


# 检测
def run():
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    MODEL_WEIGHT = 'resnet101_640x640/mask_rcnn_shapes_0040.h5'
    MODEL_WEIGHT = os.path.join(MODEL_DIR, MODEL_WEIGHT)

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(MODEL_WEIGHT, by_name=True)

    class_names = ['BG', 'pole']   #修改类别名称

    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2]

    depth_path = './myutile/depth/'

    for file_name in file_names:
        print('图片名称是：',file_name)
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
        a=datetime.now()
        # Run detection
        results = model.detect([image], verbose=1)
        b=datetime.now()
        # Visualize results
        print("time:",(b-a).seconds)
        r = results[0]
        print (r)

        #角度检测
        angles = []
        if r['rois'].shape[0]:
            for i in range(r['masks'].shape[-1]):  # 遍历每张图片中每个掩码
                mask_img=r['masks'][:,:,i].astype(int)*255
                mask_img=np.uint8(mask_img)
                img_show('mask',cv2.resize(mask_img,(800,600)))  #展示掩码
                # 掩码判断
                if cv2.countNonZero(mask_img) > 10:
                    angle = detect_angle(file_name, mask_img, depth_path)
                else:
                    angle == None
                if angle is not None:
                    angles.append(angle)
        print('angles:',angles)
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'],angles = angles)

# 角度检测函数
def detect_angle(img,mask,depth_path):
    pcs = pointProcess(img,mask,depth_path)  # 掩码转换为pcd并预处理
    infer = pointnet_inference.Pole_inference()
    points = infer.run(pcs)  # 3.pointnet预测结果  [n,6]
    print(points)
    # 角度检测
    angle = calcul_angle(points, axis='Y')
    print('检测的角度：', angle)
    return angle


if __name__ == '__main__':
    run()







