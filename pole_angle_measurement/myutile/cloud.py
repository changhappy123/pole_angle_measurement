# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:13:57 2022

@author: Administrator
"""
#点云处理文件
import open3d as o3d
from mytest import linear_fitting_3D_points
import matplotlib.pyplot as plt
import numpy as np
import pcl
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

#3.隐藏点剔除
def remove_hidden_point(p2):
    p2.paint_uniform_color([0.5, 0.5, 0.5]) #灰色
    diameter = np.linalg.norm(np.asarray(p2.get_max_bound()) - np.asarray(p2.get_min_bound()))
    camera = [0, 0, 0]       # 视点位置
    radius = diameter * 100        # 噪声点云半径,The radius of the sperical projection
    _, pt_map = p2.hidden_point_removal(camera, radius)   # 获取视点位置能看到的所有点的索引 pt_map
    
    # 可视点点云
    pcd_visible = p2.select_by_index(pt_map)
    pcd_visible.paint_uniform_color([0, 0, 1])	# 可视点为蓝色
    # 隐藏点点云
    pcd_hidden = p2.select_by_index(pt_map, invert = True)
    pcd_hidden.paint_uniform_color([1, 0, 0])	# 隐藏点为红色
    o3d.visualization.draw_geometries([pcd_visible, pcd_hidden])
    return pcd_visible

#1. 统计滤波
def reomove_statistical(p1):
    p1.paint_uniform_color([0.5, 0.5, 0.5]) #灰色
    num_neighbors = 20 # K邻域点的个数
    std_ratio = 2.0 # 标准差系数
    # 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
    sor_pcd, ind = p1.remove_statistical_outlier(num_neighbors, std_ratio)
    #o3d.visualization.draw_geometries([sor_pcd])    
    
    sor_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # 提取噪声点云
    sor_noise_pcd = p1.select_by_index(ind,invert = True)
    sor_noise_pcd.paint_uniform_color([1, 0, 0])
    # 可视化统计滤波后的点云和噪声点云
    o3d.visualization.draw_geometries([sor_pcd, sor_noise_pcd])
    return sor_pcd

#2. 聚类分割
def seg_cluster(p1):
    p1.paint_uniform_color([0.5, 0.5, 0.5]) #灰色
    eps = 0.5           # 同一聚类中最大点间距
    min_points = 50     # 有效聚类的最小点数
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(p1.cluster_dbscan(eps, min_points, print_progress=True))
    max_label = labels.max()    # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
    p1.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #o3d.visualization.draw_geometries([p1])
    
    # #点云筛选方式
    # points = np.asarray(pcd2.points)  #转换为np格式点
    # p_colors = np.asarray(pcd.colors)  #点云的颜色
    # pcd = pcd.select_by_index(np.where(points[:, 2] < y_threshold)[0]) #根据条件索引返回的点云
   
    a=np.where(colors[:,:3]!=[0,0,0])[0]
    p2 = p1.select_by_index(a)
    
    #展示
    b=np.where(colors[:,:3]==[0,0,0])[0]
    nose = p1.select_by_index(b)
    nose.paint_uniform_color([1, 0, 0])
    p2.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([nose,p2])
    
    return p2


#圆柱面拟合
def testSegment(pc):
    '''
    point_on_axis.x、point_on_axis.y、point_on_axis.z ：表示轴线上一点的x,y,z坐标
    axis_direction.x、axis_direction.y、axis_direction.z： 表示轴线的方向向量
    radius：表示圆柱体半径
    '''
    seg = pc.make_segmenter_normals(200)       # 适用于模型的最少数据个数
    seg.set_optimize_coefficients(True)       #设置对估计的模型系数需要进行优化
    seg.set_model_type(pcl.SACMODEL_CYLINDER) #设置分割模型为圆柱型
    seg.set_method_type(pcl.SAC_RANSAC)       #设置采用RANSAC作为算法的参数估计方法
    seg.set_normal_distance_weight(0.2)       #设置表面法线权重系数
    seg.set_max_iterations(10000)             #设置迭代的最大次数
    seg.set_distance_threshold(0.3)           #设置内点到模型的距离允许最大值 
    seg.set_radius_limits(0, 6)               #设置估计出圆柱模型的半径范围

    indices, model = seg.segment()            # model(7个参数)，indices 表示选择的点云索引

    #self.assertEqual(len(indices), SEGCYLIN)    
    
    return indices,model


#圆锥分割成段
def sebsection(points,d=0.5,axis='Y'):    
    if(axis=='Y'):  #如果Y轴为高度方向
        maxval=np.max(points[:,1])
        minval=np.min(points[:,1])   
    elif(axis=='X'):
        maxval=np.max(points[:,0])
        minval=np.min(points[:,0])  
        
    nums=int((maxval-minval)/d)
    
    part=[]
    for i in range(nums):
        down=minval+i*d      
        up=down+d
        #print(down,up)
        
        if(axis=='Y'):
            seg_id=(points[:,1]>=down) &(points[:,1]<up)
        elif(axis=='X'):
            seg_id=(points[:,0]>=down) &(points[:,0]<up)
            
        seg_pc=points[seg_id]       
        
        #展示：
        pcd=o3d.geometry.PointCloud()  
        pcd.points = o3d.utility.Vector3dVector(seg_pc)
        pcd.paint_uniform_color([1, 0, 0]) 
        
        pc=o3d.geometry.PointCloud()  
        pc.points = o3d.utility.Vector3dVector(points)
        pc.paint_uniform_color([0.5, 0.5, 0.5]) 
        
        #o3d.visualization.draw_geometries([pcd,pc])
        
        if len(seg_pc)>=30: #防止噪音点，点数过少
            part.append(seg_pc)
    
    return part
    

#获取中心点
def SegCone(part,nums=4,axis='Y'):
    center_pc=[]
    if len(part):
        for seg_pc in part:
            cloud=pcl.PointCloud()   # np格式转换为pcl格式
            cloud.from_array(np.array(seg_pc, dtype=np.float32))
            indices,model=testSegment(cloud)

            x,y,z,axis_x,axis_y,axis_z,=model[0:6]

            #获取中心线上的点
            if(axis=='Y'):
                maxv=np.max(seg_pc[:,1])
                minv=np.min(seg_pc[:,1])
                tmin,tmax=(minv-y)/axis_y,(maxv-y)/axis_y

            elif(axis=='X'):
                maxv=np.max(seg_pc[:,0])
                minv=np.min(seg_pc[:,0])
                tmin,tmax=(minv-x)/axis_x,(maxv-x)/axis_x

            #print('tmin:',tmin,'tmax:',tmax)
            val=(tmax-tmin)/nums

            tem=[]
            for i in range(nums+1):
                t = tmin + i * val
                x2,y2,z2=x+axis_x*t,y+axis_y*t,z+axis_z*t
                center_pc.append([x2,y2,z2])
                tem.append([x2,y2,z2])

            #展示每段中心点结果
            pc=o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(seg_pc)
            pc.paint_uniform_color([0.5, 0.5, 0.5])

            pcd=o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(tem))
            pcd.paint_uniform_color([0, 1, 0])

            #o3d.visualization.draw_geometries([pcd,pc])

    return np.array(center_pc)
            
                
#拟合中心线，计算角度
def CalculateAngle(points,axis='Y'):
    #直线拟合
    k1,b1,k2,b2=linear_fitting_3D_points(points) #最小二乘拟合空间直线
    if axis=='Y':
        maxid=np.argmax(points[:,1])
        minid=np.argmin(points[:,1])
    elif axis=='X':
        maxid=np.argmax(points[:,0])
        minid=np.argmin(points[:,0])
        
    z=points[:,-1]
    z=z[[minid,maxid]]
    x=k1*z+b1
    y=k2*z+b2
    
    endpoint=np.stack([x,y,z],axis=1)
    
    thet=np.arctan(abs((y[0]-y[-1]))/np.sqrt((np.square(x[0]-x[1])+np.square(z[0]-z[1]))))
    thet=thet*180/np.pi
    print('电杆倾斜角度为：',thet)
    
    return endpoint

#pcl拟合中心线               
def CalculateAngle2(data,axis='Y'):
    cloud=pcl.PointCloud()   # np格式转换为pcl格式
    cloud.from_array(np.array(data,dtype=np.float32))
    #cloud.from_array(data)
    
    model_p = pcl.SampleConsensusModelLine(cloud) #指定拟合点云与几何模型
    ransac = pcl.RandomSampleConsensus(model_p) #创建随机采样一致性对象
    ransac.set_DistanceThreshold(0.1) #内点到模型的最大距离
    #ransac.set_MaxIterations(1000);	 #最大迭代次数
    ransac.computeModel()  #执行RANSAC空间直线拟合
    inliers = ransac.get_Inliers() #提取内点对应的索引
    endpoint=np.vstack([cloud[inliers[0]],cloud[inliers[-1]]])
    
    x,y,z=endpoint[:,0],endpoint[:,1],endpoint[:,2]
    
    if axis=='Y':
        thet=np.arctan(abs((y[0]-y[-1]))/np.sqrt((np.square(x[0]-x[1])+np.square(z[0]-z[1]))))
    if axis=='X':
        thet=np.arctan(abs((x[0]-x[-1]))/np.sqrt((np.square(y[0]-y[1])+np.square(z[0]-z[1]))))
        
    thet=round(thet*180/np.pi,2)
    #print(thet)
    
    return endpoint,inliers,thet
    
#可视化直线
def drawLine(endpoint):
    lines = [[0, 1]] #连接的顺序，封闭链接
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector([[1,0,0]]) #线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(endpoint)

    return lines_pcd


def detect(path,axis='X'):
    path=os.path.join(BASE_DIR,path)
    pc= o3d.io.read_point_cloud(path)
    #pc= o3d.geometry.PointCloud.uniform_down_sample(pc, 5) #下采样   
    #点云缩放
    #pc = pc.scale(1,(0,0,0))    
    
    points=np.asarray(pc.points)
    
    #减均值：
    # pcmean=np.mean(points.T,axis=1)
    # pcmean=np.expand_dims(pcmean,axis=1)
    # points=(points.T-pcmean).T
    
    #归一化
    # centroid = np.mean(points, axis=0)
    # points = points - centroid
    # m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    # points = points / m
    
    part=sebsection(points,d=3,axis=axis)
    center=SegCone(part,nums=5, axis=axis)
    #endpoint= CalculateAngle(center,axis='X')
    endpoint,inliers,thet = CalculateAngle2(center,axis=axis)
    lines_pcd=drawLine(endpoint) #中心线
    
    pc.paint_uniform_color([0.5, 0.5, 0.5]) 
    
    
    pcd=o3d.geometry.PointCloud()  #中心点
    pcd.points = o3d.utility.Vector3dVector(center)
    pcd.paint_uniform_color([0, 1, 0]) 
    
    o3d.visualization.draw_geometries([pc,lines_pcd,pcd])
    
    return points


#################################
# 预测，中轴线拟合，角度计算
#################################
def calcul_angle(points, axis='X'):
    thet = None
    if not len(points):
        return thet
    points = points[:,:3]
    part = sebsection(points, d=1, axis=axis)
    center = SegCone(part, nums=5, axis=axis)
    # endpoint= CalculateAngle(center,axis='X')
    print('中心点：',center)

    if len(center):
        endpoint, inliers,thet = CalculateAngle2(center, axis=axis)
        lines_pcd = drawLine(endpoint)  # 中心线

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.paint_uniform_color([0.5, 0.5, 0.5])

        pcd = o3d.geometry.PointCloud()  # 中心点
        pcd.points = o3d.utility.Vector3dVector(center)
        pcd.paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries([pc, lines_pcd, pcd])

    return thet


#测试
if __name__=='__main__':    
    #整体测试
    # path='E:/linux_data/utile/experimental/tuyang_exp/pre_pro.pcd'
    # points=detect(path,axis='X')

    #p ='E:/linux_data/utile/apollo'
    p ='E:/linux_data/utile/models/rm_pcd3D'
    files=os.listdir(p)
    for path in files:
        path=os.path.join(BASE_DIR,p,path)
        points=detect(path,axis='X')
