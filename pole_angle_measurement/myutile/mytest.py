# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:43:08 2022

@author: Administrator
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def img_show(img):
    cv2.imshow('result',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#############################
#加载相机参数
############################
#相机内参：
data_config= {
    'Camera_5': np.array(
        [2304.54786556982, 2305.875668062,
         1686.23787612802, 1354.98486439791]),
    'Camera_6': np.array(
        [2300.39065314361, 2301.31478860597,
         1713.21615190657, 1342.91100799715])}

#从图片找到对应pose文件,返回pose路径
def from_prePath_find_posefile(img_path,tar_path,root_path):
    '''
    输入：
        root_path: pose文件所在根目录 如：'F:\\RGBImage'
        img_path: 图片名称，例如 170908_061436555_Camera_5.jpg
        tar_path: 带有目标路径的文件 target_RGB_img.txt        
    返回： 
       对应  pose.txt 文件路径
    '''
    with open(tar_path,'r') as f:
        paths=[x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        
    all_pose_path={}
    for path in paths:
        path=(root_path+path).replace('ColorImage','Pose')
        p,img_name=os.path.split(path)
        pose_path=os.path.join(p,'pose.txt')
        all_pose_path.update({img_name:pose_path})
    
    if img_path in all_pose_path.keys():
        return all_pose_path[img_path]


#从pose文件找到对应图片的相机外参
'''
读取文件：
1. read([size])方法从文件当前位置起读取size个字节，若无参数size，
   则表示读取至文件结束为止，它范围为字符串对象
  with open(path,'r') as f:
      sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
2. readline()从字面意思可以看出，该方法每次读出一行内容，所以，
   读取时占用内存小，比较适合大文件，该方法返回一个字符串对象
  with open(path,'r') as f:
      file=f.readline()
3. readlines()方法读取整个文件所有行，保存在一个列表(list)变量中，
   每行作为一个元素，但读取大文件会比较占内存      
4. poses = [line for line in open(pose_file_in)]  
'''
def from_lastPath_find_pose(pose_path,img_path):
    '''
    输入：
        pose_path: pose文件路径
        img_path: 图片名称 如：170908_061436555_Camera_5.jpg
    返回：
        R： [3,3]
        T： [3]相机外参
    '''
    with open(pose_path,'r') as f:
        poses=[x.strip() for x in f.read().strip().splitlines() if len(x.strip())]

    parameter={}
    for pose in poses:
        pose=pose.split(' ')
        img_name=pose[-1]
        mat=[np.float32(num.strip()) for num in pose[:-1]]
        mat=np.array(mat).reshape((4,4))
        parameter[img_name]=mat
    
    if img_path in parameter.keys():
        R=parameter[img_path][:3,:3]
        T=parameter[img_path][:3,-1]
        return R,T

#获取每张图片对应的内参
def img_name_to_intrinsic(img):
    '''
    输入： 图片名称 如：170908_061436555_Camera_5.jpg
           img_raw_shape 原始图片尺寸 （h，w）
           img_shape resize后的尺寸（h，w）
    返回： 对应相机内参 [4,] 
    '''
    intrinsic=data_config[img[-12:-4]] #原始内参
    # #内参归一化
    # if img_shape:
    #     intrinsic[[0,2]]=intrinsic[[0,2]]/img_raw_shape[1]*img_shape[1]
    #     intrinsic[[1,3]]=intrinsic[[1,3]]/img_raw_shape[0]*img_shape[0]
    return intrinsic
    
    
############################
#相机坐标转换
############################
#将内参参数转换为内参矩阵
def intrinsic_vec_to_mat(intrinsic,scale=None):
    """Convert a 4 dim intrinsic vector to a 3x3 intrinsic
       matrix （fx，fy，cx，cy）
    """
    
    if scale is None:
        scale = [1, 1]

    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = intrinsic[0] * scale[1]
    K[1, 1] = intrinsic[1] * scale[0]
    K[0, 2] = intrinsic[2] * scale[1]
    K[1, 2] = intrinsic[3] * scale[0]
    K[2, 2] = 1.0
    return K

#世界坐标系转换为相机坐标  C=(R*P+T)  或 Rt*P
def world_to_camera_frame(P, R, T):
  """
  Convert points from world to camera coordinates

  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 3d points in camera coordinates
  """
  assert len(P.shape) == 2
  assert P.shape[1] == 3
  return (np.dot(R, P.T) + T.reshape(-1,1)).T

#相机坐标转换为像素坐标 Pix=(I*C)/Zc 也即[fx*Xc+u*Zc,fy*Yc+v*Zc,Zc]/Zc
def camera_to_pixel(I,P):
    '''
    I: 内参矩阵 [3,3]
    P: 相机坐标 [n,3]
    返回 （u，v，d） [n,3]  d为绝对深度
    '''
    cam=np.dot(I,P.T).T
    cam[...,:-1]=cam[...,:-1]/cam[...,-1].reshape(-1,1)
    return cam

#读取深度图，返回绝对距离
def read_depth_img(depth_path,scale=None):
    #depth_img=cv2.imread(depth_path, cv2.CV_16UC1)
    depth_img=cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)  #读取原始深度
    if scale is not None:
        w,h=int(depth_img.shape[1]/scale[1]),int(depth_img.shape[0]/scale[0])
        depth_img=cv2.resize(depth_img,(w,h))
    max_value=2**(int(str(depth_img.dtype)[-2:]))/200
    depth_img=depth_img/200  #绝对深度（m）
    return depth_img,max_value

# 获取掩码像素点或全图像素点
def get_piex(img,entire=False,*args):
    # args: 图片的宽高(h,w)
    # 返回： 像素点位置(h,w)  维度：[n,2]
    if entire:
        #全图像素点
        h=np.arange(args[0])  #h
        w=np.arange(args[1])  #w
        mask1,mask2=np.meshgrid(h,w)
        mask_index=list(zip(mask1.flatten(),mask2.flatten()))
        mask_index=np.array(mask_index)
        return mask_index   
    #否则返回掩码像素点
    mask_index=np.where(img[...,2]!=0)
    #mask_index=list(map(list, zip(*mask_index)))
    mask_index=np.array(list(zip(*mask_index)))  #掩码像素点
    return mask_index

#像素坐标和深度转换为相机坐标
def pixel_to_camera_frame(Piex,mat,depth):
    # mat为内参矩阵[3,3] Piex为像素坐标[n,2] depth为深度[n,]
    # 返回相机坐标 [n,3]
    new_Piex=np.zeros((Piex.shape[0],3))
    new_Piex[:,:-1]=Piex
    new_Piex[:,-1]=1  #像素坐标转换为其次坐标 [n,3]
    return (np.dot(np.linalg.inv(mat),new_Piex.T)*depth).T

#相机坐标转换为世界坐标
def camera_to_world_frame2(P, R, T):
    '''
    Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
   Returns
    X_cam: Nx3 points in world coordinates
    '''
    RT=np.zeros((4,4))
    RT[:-1,:-1]=R
    RT[:-1,-1]=T
    RT[-1,-1]=1
    a=np.ones(P.shape[0])
    P=np.hstack((P,a.reshape(-1,1)))   
    joint=(np.dot(np.linalg.inv(RT),P.T)).T
    return joint[...,:-1]


#读取标注文件生成对应二维关键点
def read_label_file(json_path):
    '''
    返回对应图片关键点二维信息
    '''
    pass

#将二维关键点转换为三维关键点
def convert_2D_to_3D(pixel,depth_path,C,R=None,T=None,scale=None):
    '''
   input: 
     pixel : 像素点 [n,2]    (v,u)
     depth_path： 对应深度图路径
     C ：内参矩阵 [4]
     R :旋转矩阵 [3,3]
     T :平移矩阵 [3,]
     img_shape: resize后的尺寸 (h,w)
   return:
     point ：实际坐标 [n,3]   (x,y,z)
    '''
    depth,max_value=read_depth_img(depth_path,scale=None) #深度图
    #pixel=list(map(list,list(pixel))) #将像素位置数组转换为列表
    depth=depth[list(pixel[:,0]),list(pixel[:,1])] # 对应深度值 [n,]
    mat=intrinsic_vec_to_mat(C,scale=None)  #转换为内参矩阵
    #去除深度最大值的深度和对应的像素点
    # removed_value=np.max(depth)
    # if round(removed_value)==round(max_value):
    #     removed_id=np.where(depth!=removed_value)[0]
    #     depth=depth[removed_id]
    #     pixel=pixel[removed_id]
    
    #将像素点（h，w）转换为（w，h）
    #np.stack(array,axis=1)  axis=num 表示取每个数组的第num维的数组进行重新堆叠
    pixel=np.stack([pixel[:,-1],pixel[:,0]],axis=1)

    point=pixel_to_camera_frame(pixel,mat,depth) #像素点转换为相机坐标点
    #point=camera_to_world_frame2(point, R, T)   #相机坐标点转换为世界坐标点
    return point


def visualization(points,isDrawLine=False,*args):
    '''
    #绘制散点图
    ax.scatter(xs, ys, zs, s=20, c=None, depthshade=True, *args, *kwargs)
    xs,ys,zs：输入数据；
    s:scatter点的尺寸
    c:颜色，如c = 'r’就是红色；
    depthshase:透明化，True为透明，默认为True，False为不透明
    *args等为扩展变量，如maker = ‘o’，则scatter结果为’o‘的形状
    '''
    #绘制点云图
    plt.figure("3D Scatter", facecolor="lightgray")
    ax3d = plt.gca(projection="3d")  # 创建三维坐标
    #点坐标
    x=points[:,0]
    y=points[:,1]
    z=points[:,2]
    
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2) # 点大小
    plt.title('3D Scatter', fontsize=20)
    ax3d.set_xlabel('x', fontsize=14)
    ax3d.set_ylabel('y', fontsize=14)
    ax3d.set_zlabel('z', fontsize=14)
    plt.tick_params(labelsize=10)
    # ax3d.set_xlim(550,565)
    # ax3d.set_ylim(19,34)
    # ax3d.set_zlim(400,415)
    #ax3d.scatter(x, y, z, s=20, c=d, cmap="jet", marker="o")
    ax3d.plot(x,y,z, 'b.', markersize=0.2)
    #绘制拟合直线
    if isDrawLine:  
        #print(args)
        ax3d.plot(args[0],args[1], args[2],'r', label='parametric curve')
    ax3d.legend()
    plt.show()
    
    
##  由空间3维点拟合出一条直线
def linear_fitting_3D_points(points):
    '''
    用直线拟合三维空间数据点。
    参考https://www.doc88.com/p-8189740853644.html  
    x = k1 * z + b1
    y = k2 * z + b2
    Input:
        points    ---   List， 三维空间数据点，例如：
                        [[2,3,48],[4,5,50],[5,7,51]]
                    
    返回值是公式系数 k1, b1, k2, b2
    '''
    #表示矩阵中的值
    Sum_X=0.0
    Sum_Y=0.0
    Sum_Z=0.0
    Sum_XZ=0.0
    Sum_YZ=0.0
    Sum_Z2=0.0
 
    for i in range(0,len(points)):
        xi=points[i][0]
        yi=points[i][1]
        zi=points[i][2]
 
        Sum_X = Sum_X + xi
        Sum_Y = Sum_Y + yi
        Sum_Z = Sum_Z + zi
        Sum_XZ = Sum_XZ + xi*zi
        Sum_YZ = Sum_YZ + yi*zi
        Sum_Z2 = Sum_Z2 + zi**2
 
    n = len(points) # 点数
    den = n*Sum_Z2 - Sum_Z * Sum_Z # 公式分母
    k1 = (n*Sum_XZ - Sum_X * Sum_Z)/ den
    b1 = (Sum_X - k1 * Sum_Z)/n
    k2 = (n*Sum_YZ - Sum_Y * Sum_Z)/ den
    b2 = (Sum_Y - k2 * Sum_Z)/n
    
    return k1, b1, k2, b2  

#%%
if __name__=='__main__':
    #测试 1 apollo测试
    img_path='170908_082025654_Camera_5.jpg'   # 图片名称
    path='E:/linux_data/utile/models/label/cv2_mask/' +img_path[:-4]+'.png'  #掩码图片路径                    
    depth_path='F:\\pole_tar/tar_depth/'+img_path[:-4]+'.png' #对应深度图路径
    #depth_path='3D_Coordinate/171206_061831005_Camera_5.png'
    root_path='F:\\RGBImage'  #根目录
    tar_path='F:\\pole_tar/target_RGB_img.txt' #含有电杆目标图片的路径文件
    
    #复制深度图到当前文件
    # import shutil
    # shutil.copy(depth_path,'E:/linux_data/utile/3D_Coordinate')

    #获取电杆掩码
    img=cv2.imread(path)
    #img_show(img)


    #获取掩码像素点,或全图像素点
    mask_index=get_piex(img)  #掩码像素点
    #mask_index=get_piex(img,True,*img.shape[:2]) #全图像素点
   

    #每个像素点的颜色
    colors=cv2.imread('F:\\pole_tar/tar_img/'+img_path)
    colors=colors[list(mask_index[:,0]),list(mask_index[:,1])]/[255,255,255]
    
    #读取相机内外参数
    pose_path=from_prePath_find_posefile(img_path,tar_path,root_path) #每张图片对应相机外参 pose 路径
    #pose_path='3D_Coordinate/pose.txt'
    R,T=from_lastPath_find_pose(pose_path,img_path) #相机外参
    C=img_name_to_intrinsic(img_path)  #相机原始内参
    
    #获取关键点三维信息
    points=convert_2D_to_3D(mask_index,depth_path,C,R,T)
    #visualization(points)


    #可视化，将点云转换成open3d中的数据形式并用pcd来保存，以方便用open3d处理
    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])#按顺序全部赋色
    o3d.visualization.draw_geometries([pcd])

    #o3d.io.write_point_cloud('E:/linux_data/utile/experimental/res/whole_pc/'+img_path[:-4]+'_c.pcd', pcd, True)
    
    
    
#%%
    #读取文件
    pcd = o3d.io.read_point_cloud("tuyang/depth_pole10.pcd")
    o3d.visualization.draw_geometries([pcd])
    points= np.asarray(pcd.points)
    # visualization(points)
    len(pcd.points)
    
    
    pcd.paint_uniform_color([0.5, 0.5, 0.5]) #全部设为灰色
    #pcd.colors[1500] = [0.5, 0, 0.5] # 将第1500个点设置为紫色 
    #全图点云赋色
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])#按顺序全部赋色
    o3d.visualization.draw_geometries([pcd])
    
    
    #掩码全部赋色
    pcd2 = o3d.io.read_point_cloud("pcd/mask_pole.pcd")
    pcd2.paint_uniform_color([0.5, 0.5, 0.5]) 
    color=cv2.imread('image/171206_061831005_Camera_5_result.png')
    mask=get_piex(img)  #掩码像素点
    color=color[list(mask[:,0]),list(mask[:,1])]/[255,255,255]
    pcd2.colors = o3d.utility.Vector3dVector(color[:, :3])#按顺序全部赋色
    o3d.visualization.draw_geometries([pcd2])
    
 
#%%    
    #展示点云
    # pcd = o3d.io.read_point_cloud("models/pcd3D/22/170927_074137694_Camera_6_2.pcd")
    # o3d.visualization.draw_geometries([pcd])
    # points= np.asarray(pcd.points)
    
    #visualization(points)
    # pcd=o3d.geometry.PointCloud()  #中心点
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.io.write_point_cloud('tuyang01.pcd', pcd, True)
    def vis_draw_geometries2(pcds):
        vis = o3d.visualization.Visualizer()
        vis.create_window()	#创建窗口
        render_option: o3d.visualization.RenderOption = vis.get_render_option()	#设置点云渲染参数
        render_option.background_color = np.array([0, 0, 0])	#设置背景色（这里为黑色）
        render_option.point_size = 2	#设置渲染点的大小
        for pcd in pcds:
            vis.add_geometry(pcd)	#添加点云
        vis.run()
        
    #测试2：图漾相机
    C=[599.015869140625, 0.0, 330.419921875, 0.0, 599.015869140625, 255.3875274658203, 0.0, 0.0, 1.0]
    mat=np.array(C).reshape(3,3)
    
    w,h=640,480
    a=np.arange(h)
    b=np.arange(w)
    mask1,mask2=np.meshgrid(a,b)
    mask_index=list(zip(mask1.flatten(),mask2.flatten()))
    pixel=np.array(mask_index)
    
    img=cv2.imread('E:/linux_data/utile/experimental/tuyang_exp/label.png')
    mask_index=np.where(img[...,2]!=0)
    pixel=np.array(list(zip(*mask_index)))  #掩码像素点
    
    depth_path='E:/linux_data/utile/experimental/tuyang_exp/2022-09-20T21-50-58_100-100.png'
    depth=cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) 
    depth=depth[list(pixel[:,0]),list(pixel[:,1])] # 对应深度值 [n,]
    
    pixel=np.stack([pixel[:,-1],pixel[:,0]],axis=1)
    points=pixel_to_camera_frame(pixel,mat,depth) #像素点转换为相机坐标点
    
    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, 5)
    
    vis_draw_geometries2([pcd_new])
    #o3d.visualization.draw_geometries([pcd_new])
    
    o3d.io.write_point_cloud('./tuyang01.pcd', pcd, True)
    #points2pcd(points,'depth_pole1')

    









































