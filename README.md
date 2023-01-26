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
Download pre-trained COCO weights ([mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases))  
Download [ApolloScape](https://apolloscape.auto/) datasets  
The image segmentation and point cloud segmentation models in this model are trained separately.    
When training the image segmentation model, the training data set folder needs to be provided. The pictures are labeled with Labelme software, and the file format after labeling is as follows:  
|--label  
|    |----cv2_mask  
|    |----json  
|    |----labelme_json  
|    |----pic  
When training the point cloud segmentation model, download the prepared HDF5 file or prepare it yourself, then in pole_ angle_ measurement/pole_pointnet file start the training:  
     python train.py --log_dir log2 --test_area 2

# Evaluation
Prepare pre-training parameter file（mask_rcnn_box/logs/mask_rcnn_shapes_0040.h5 and pole_pointnet/log2/model.ckpt） to test：  
     python inference.py  
# References
[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation by Qi et al. (CVPR 2017 Oral Presentation)](http://stanford.edu/~rqi/pointnet/).   
[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space by Qi et al. (NIPS 2017).](https://proceedings.neurips.cc/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html)   
[Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)  
[Frustum PointNet: Frustum PointNets for 3D Object Detection from RGB-D Data](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Frustum_PointNets_for_CVPR_2018_paper.pdf)  
[ApolloScape: The ApolloScape Dataset for Autonomous Driving](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w14/html/Huang_The_ApolloScape_Dataset_CVPR_2018_paper.html)


