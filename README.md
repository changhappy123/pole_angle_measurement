# pole_angle_measurement
Automatic measurement of inclination angle of utility poles using 2D image and 3D point cloud.
![show](https://user-images.githubusercontent.com/87463009/214590473-7c317338-13c2-49f3-b85b-d7b5ac819b88.jpg)



# Introduction
The utility pole inclination angle is an important parameter for determining the pole health conditions. Without depth information, the angel cannot be estimated from 2D image and without large labeled reference pole data, to locate the pole in the 3D point cloud is time consuming. Therefore, this paper proposed a method that processes the pole data from the 2D image and 3D point cloud to automatically measure the pole inclination angle. Firstly, the mask of the pole skeleton is obtained from an improved Mask RCNN. Secondly, the pole point cloud is extracted from a PointNet which deals with the generated frustum from the pole skeleton mask and depth map fusion. Finally, the angle is calculated by fitting the central axis of the pole cloud data. ApolloSpace open dataset and laboratory data are used for evaluation.It is proved that the method can effectively realize the automatic measurement of pole inclination.
# Installation
This is an implementation of utility pole angle measurement on Python 3, Keras2.0.8, and TensorFlow1.15.0. There are also some dependencies for a few Python libraries for data processing and visualizations like cv2, open3d,pypcl etc. It's highly recommended that you have access to GPUs.
# Usage
At present, we support the training and testing of the model, as well as the detection results of the 2d detector, the segmentation results of the 3D detector and the tilt angle in the camera coordinate system (under the apollospace dataset).
# Training
Download pre-trained COCO weights (mask_rcnn_coco.h5)


# Evaluation
Download pre-trained COCO weights (mask_rcnn_coco.h5)


# References
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation by Qi et al. (CVPR 2017 Oral Presentation).   
PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space by Qi et al. (NIPS 2017).   
Mask R-CNN  
Frustum PointNet: Frustum PointNets for 3D Object Detection from RGB-D Data  
ApolloSpace: The ApolloScape Dataset for Autonomous Driving


