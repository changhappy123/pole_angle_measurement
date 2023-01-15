# -*- coding: utf-8 -*-
'''
获取掩码对应的pcd文件，用于训练pointnet模型
'''

import os
import json
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from mytest import *
import cloud


# np.expand_dims(img_mask,2).repeat(3,axis=2) #np某维度进行复制
def img_show(winname,img):
    cv2.imshow(winname,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#添加隐藏点区分目标
def remove_hidden_point2(p2):
    p2.paint_uniform_color([0.5, 0.5, 0.5]) #灰色
    diameter = np.linalg.norm(np.asarray(p2.get_max_bound()) - np.asarray(p2.get_min_bound()))
    camera = [0, 0, 0]       # 视点位置
    radius = diameter * 100        # 噪声点云半径,The radius of the sperical projection
    _, pt_map = p2.hidden_point_removal(camera, radius)   # 获取视点位置能看到的所有点的索引 pt_map
   
    color=np.asarray(p2.colors)
    color[pt_map]=[0,0,1]
    p2.colors = o3d.utility.Vector3dVector(color)
    #展示：
    o3d.visualization.draw_geometries([p2])
    return p2


# 统计滤波
def reomove_statistical_2(p1):
    #p1.paint_uniform_color([0.5, 0.5, 0.5]) #灰色
    num_neighbors = 20 # K邻域点的个数
    std_ratio = 2.0 # 标准差乘数
    # 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
    sor_pcd, ind = p1.remove_statistical_outlier(num_neighbors, std_ratio)
    #o3d.visualization.draw_geometries([sor_pcd])    
    
    #sor_pcd.paint_uniform_color([0, 0, 1])
    # 提取噪声点云
    sor_noise_pcd = p1.select_by_index(ind,invert = True)
    sor_noise_pcd.paint_uniform_color([1, 0, 0])
    # 可视化统计滤波后的点云和噪声点云
    #o3d.visualization.draw_geometries([sor_pcd, sor_noise_pcd])
    return sor_pcd


# 聚类分割
def seg_cluster_2(p1):
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
    #o3d.visualization.draw_geometries([p2])
    return p2


#深度图和掩码生成pcd点云
def generate_pcd():
    path='E:\\linux_data\\utile\\3D_Coordinate\\170908_085030315_Camera_5.png'
    img=cv2.imread(path)
    h,w=img.shape[:2]
   
    depth_root='F:\\pole_tar/tar_depth/'
    json_path='../models/label/json'
    json_files=os.listdir(json_path)
    
    for json_file in json_files:
        #读取json文件
        jsonfile=os.path.join(json_path,json_file)
        with open(jsonfile,'r') as file:
            json_dict = json.load(file)
            
        num_obj=len(json_dict['shapes'])  #目标个数
        mask = np.zeros([h, w, 3], dtype=np.uint8) #对应掩码
    
        img_name=jsonfile[-30:-5]
        img_path=img_name+'.jpg'
        
        for ids in range(num_obj):
            point=np.array(json_dict['shapes'][ids]['points'])
            point=point.astype(np.int)
            #1. 随机掩码改变质心和大小
            
            # 生成2D点和3D点
            re=cv2.drawContours(mask.copy(),[point],0,(255,255,255),-1) #掩码
            # cv2.imwrite('01.jpg',re)
            # sre=cv2.resize(re,(256,256))
            # img_show(sre)
            
            mask_index=get_piex(re)  #2D坐标点
    
            #pose_path=from_prePath_find_posefile(img_path,tar_path,root_path) #每张图片对应相机外参 pose 路径
            #R,T=from_lastPath_find_pose(pose_path,img_path) #相机外参
    
            depth_path = depth_root + img_name + '.png'
            C=img_name_to_intrinsic(img_path)  #相机原始内参
            np_points=convert_2D_to_3D(mask_index,depth_path,C) #3D点
            
            #转换为点云处理
            pcd=o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_points)

            p2=cloud.reomove_statistical(pcd) #统计滤波
            p3=cloud.seg_cluster(p2)  #聚类分割        
            p4=remove_hidden_point2(p3)   #添加隐藏点区分目标
            pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, 5)  #均匀下采样
            
            o3d.io.write_point_cloud('./pcd3D/'+img_name+'_'+str(ids+1)+'.pcd', pcd_new, True)


#图漾相机获取2D点和3D点
def get_mask(img,entire=False):
    if not entire: #掩码
        mask_index=np.where(img[...,2]!=0)
        mask_index=np.array(list(zip(*mask_index)))  #掩码像素点
    else:
        w,h=640,480
        a=np.arange(h)
        b=np.arange(w)
        mask1,mask2=np.meshgrid(a,b)
        mask_index=list(zip(mask1.flatten(),mask2.flatten()))
        pixel=np.array(mask_index)
    
    return pixel


def get_3D_points(pixel,depth_path):
    
    C=[599.015869140625, 0.0, 330.419921875, 0.0, 599.015869140625, 255.3875274658203, 0.0, 0.0, 1.0]
    mat=np.array(C).reshape(3,3)
    
    depth=cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) 
    depth=depth[list(pixel[:,0]),list(pixel[:,1])] # 对应深度值 [n,]
    
    pixel=np.stack([pixel[:,-1],pixel[:,0]],axis=1)
    points=pixel_to_camera_frame(pixel,mat,depth) #像素点转换为相机坐标点
    return points


def gen_pcd():
    depth_root='E:\\linux_data/utile/models/result/pic/'
    json_path='E:\\linux_data/utile/models/result/json'
    json_files=os.listdir(json_path)
    w,h=640,480
    
    for json_file in json_files:
        #读取json文件
        jsonfile=os.path.join(json_path,json_file)
        with open(jsonfile,'r') as file:
            json_dict = json.load(file)
            
        num_obj=len(json_dict['shapes'])  #目标个数
        mask = np.zeros([h, w, 3], dtype=np.uint8) #对应掩码
    
        img_name=json_file[:-5]
        img_path=img_name+'.jpg'
        
        for ids in range(num_obj):
            point=np.array(json_dict['shapes'][ids]['points'])
            point=point.astype(np.int)
            #1. 随机掩码改变质心和大小
            
            # 生成2D点和3D点
            re=cv2.drawContours(mask.copy(),[point],0,(255,255,255),-1) #掩码
            cv2.imwrite('01.jpg',re)
            sre=cv2.resize(re,(256,256))
            #img_show(sre)
            
            mask_index=get_piex(re)  #2D坐标点
    
            depth_path = depth_root + img_name + '.png'
            np_points=get_3D_points(mask_index,depth_path)
            
            #转换为点云处理
            pcd=o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_points)
            #o3d.visualization.draw_geometries([pcd])
            
            p2=cloud.reomove_statistical(pcd) #统计滤波
            p3=cloud.seg_cluster(p2)  #聚类分割        
            p4=remove_hidden_point2(p3)   #添加隐藏点区分目标
            pcd_new = o3d.geometry.PointCloud.uniform_down_sample(p4, 5)  #均匀下采样
            
            o3d.io.write_point_cloud('./rm_pcd3D/'+img_name+'_'+str(ids+1)+'.pcd', pcd_new, True)

            
##################################################
#论文图标展示
####################################################
#掩码随机变换
def mask_enhance(box2d,img_path,augmentX=3,perturb_box2d=True,vis=True):
    #进行几次 2Dbox 增强
    for _ in range(augmentX): 
        if perturb_box2d:
            new_mask,center,new_center = random_shift_box2d(box2d)
            print(box2d)
            print(new_mask)
            
            if vis:
                #展示：
                # mask=cv2.imread(img_path)
                # re=cv2.drawContours(mask.copy(),[box2d],0,(255,0,0),-1) #掩码
                # sre=cv2.resize(re,(800,600))
                # img_show('原始掩码：',sre)
                # modify=cv2.drawContours(mask.copy(),[new_mask],0,(0,255,0),-1) #掩码
                # mre=cv2.resize(modify,(800,600))
                # img_show('随机掩码：',mre)
                
                res=cv2.imread(img_path)
                #res=mask[800:2100,2500:3000]
                               
                cv2.drawContours(res,[new_mask],0,(0,125,200),-1) #掩码
                cv2.drawContours(res,[box2d],0,(255,0,0),3) #掩码  
                
                for i in range(len(new_mask)):
                    cv2.line(res, new_center[0], new_mask[i],[0,0,255], 1)
                    
                cv2.circle(res, center[0], 3, (255, 0, 0), 3)
                cv2.circle(res, new_center[0], 3, (0, 0, 255), 3)
                #mre=cv2.resize(res,(200,800))
                res=res[800:2100,2400:2900]
                img_show('掩码随机变换：',res)
                cv2.imwrite('E:/linux_data/utile/123.jpg',res)
        else:
            xmin,ymin,xmax,ymax = box2d


def random_shift_box2d(box2d, shift_ratio=0.1):
    r = shift_ratio
    cx = np.mean(box2d[:,0])  #中心点
    cy = np.mean(box2d[:,1])
    
    center=np.tile(np.array([cx,cy]),(4,1))
    direction=np.where((box2d-center)>0,1,-1) #方向
    
    #中心点、距离和斜率变化
    dis=np.sqrt(np.sum((center-box2d)**2,axis=1))
    dis2 = dis*(1+np.random.random()*2*r-r) # 0.9 to 1.1    
    #dis2 = dis*np.sqrt(1+np.random.random()*2*r-r) # 0.9 to 1.1
    
    
    slope = (box2d-center)[:,1]/(box2d-center)[:,0]
    slope2 = slope*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    #slope2 = slope* np.sqrt((1+np.random.random()*2*r-r))
    
    x= sorted(box2d[:,0])
    y= sorted(box2d[:,1])
    w= (x[2]+x[3])/2 - (x[0]+x[1])/2
    h= (y[2]+y[3])/2 - (y[0]+y[1])/2
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)

    
    #计算新坐标值
    # orix = np.sqrt(dis**2/(slope**2+1))+center[:,0]
    # # oriy= center[:,1] + slope*(orix-center[:,0])
    # oriy = np.sqrt(dis**2/((1/slope**2)+1))+center[:,1]
    # oribox = np.stack([orix,oriy],axis=1)
    
    new_center = np.tile(np.array([cx2,cy2]),(4,1))
    px = np.sqrt(dis2**2/(slope2**2+1))*direction[:,0]+new_center[:,0]
    py = new_center[:,1] + slope2*(px-new_center[:,0])
    
    #py= slope2*px+(new_center[:,1]-slope2*new_center[:,0])
    
    result = np.stack([px,py],axis=1)
    result=np.asarray(result,dtype=int)
    
    return result, np.asarray(center,dtype=int), np.asarray(new_center,dtype=int)


#################################
#展示预处理过程
#################################
def showProcess():    
    #单张测试，实验展示
    depth_root='E:/linux_data/utile/experimental/depth/'
    img_root='E:/linux_data/utile/experimental/img/'
    jsonfile='../experimental/label/json/171206_054051610_Camera_5.json'
    
    with open(jsonfile,'r') as file:
        json_dict = json.load(file)
        
    h,w=2710,3384
    num_obj=len(json_dict['shapes'])  #目标个数
    mask = np.zeros([h, w, 3], dtype=np.uint8) #对应掩码
    
    img_name=jsonfile[-30:-5]
    img_path=img_name+'.jpg'
    imgPath=img_root+img_path
    
    for ids in range(num_obj):
        point=np.array(json_dict['shapes'][ids]['points'])
        point=point.astype(np.int)
         
        #mask_enhance(point,imgPath)
        
        # 生成2D点和3D点
        re=cv2.drawContours(mask.copy(),[point],0,(255,255,255),-1) #掩码
        cv2.imwrite('01.jpg',re)
        sre=cv2.resize(re,(256,256))
        img_show('result',sre)
        #cv2.imwrite('E:/linux_data/utile/experimental/mask.png',re)
        
        #1. 随机掩码改变质心和大小
        # img_show(change_re)
        
        mask_index=get_piex(re)  #2D坐标点
    
        #pose_path=from_prePath_find_posefile(img_path,tar_path,root_path) #每张图片对应相机外参 pose 路径
        #R,T=from_lastPath_find_pose(pose_path,img_path) #相机外参
    
        depth_path = depth_root + img_name + '.png'
        C=img_name_to_intrinsic(img_path)  #相机原始内参
        np_points=convert_2D_to_3D(mask_index,depth_path,C) #3D点
        
        #转换为点云处理
        pcd=o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_points)
        pcd.paint_uniform_color([0.5,0.5,0.5])  # 全部设为绿色
        o3d.visualization.draw_geometries([pcd],width=800,height=600)
        
        p2=cloud.reomove_statistical(pcd) #统计滤波
        p3=cloud.seg_cluster(p2)  #聚类分割        
        p4=remove_hidden_point2(p3)   #添加隐藏点区分目标
        
        pcd_new = o3d.geometry.PointCloud.uniform_down_sample(p4, 5)  #均匀下采样
        o3d.visualization.draw_geometries([pcd_new],width=800,height=600)
        #o3d.io.write_point_cloud('./pcd3D/'+img_name+'_'+str(ids+1)+'.pcd', pcd_new, True)

#################################
#预测掩码生成pcd文件
#################################
def showPreProcess(img_path):
    import glob
    #img_path='170908_062231135_Camera_6.jpg'   # 图片名称
    maskroot='E:/linux_data/utile/experimental/res/mask/' #掩码图片路径                    
    depth_path='E:/linux_data/utile/experimental/res/depth/'+img_path[:-4]+'.png' #对应深度图路径
    root_path='F:\\RGBImage'  #根目录
    tar_path='F:\\pole_tar/target_RGB_img.txt' #含有电杆目标图片的路径文件
    
    masklist=glob.glob(maskroot+img_path[:-4]+'_*.png')
    
    for ids in range(len(masklist)):
        #获取电杆掩码
        img=cv2.imread(masklist[ids])
        #img_show(img)
       
        #获取掩码像素点,或全图像素点
        mask_index=get_piex(img)  #掩码像素点
        #mask_index=get_piex(img,True,*img.shape[:2]) #全图像素点
       
        #每个像素点的颜色
        # colors=cv2.imread('F:\\pole_tar/tar_img/'+img_path)
        # colors=colors[list(mask_index[:,0]),list(mask_index[:,1])]/[255,255,255]
        
        #读取相机内外参数
        #pose_path=from_prePath_find_posefile(img_path,tar_path,root_path) #每张图片对应相机外参 pose 路径
        #R,T=from_lastPath_find_pose(pose_path,img_path) #相机外参
        C=img_name_to_intrinsic(img_path)  #相机原始内参
        
        #获取关键点三维信息
        points=convert_2D_to_3D(mask_index,depth_path,C)
    
        #可视化，将点云转换成open3d中的数据形式并用pcd来保存，以方便用open3d处理
        pcd=o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        #pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])#按顺序全部赋色
        o3d.visualization.draw_geometries([pcd])
        
        #o3d.io.write_point_cloud('E:/linux_data/utile/experimental/res/pcd/'+img_path[:-4]+'_'+str(ids+1)+'.pcd', pcd, True)    
        
        #预处理：
        pcd.paint_uniform_color([0.5,0.5,0.5])  # 全部设为绿色
        p2=cloud.reomove_statistical(pcd) #统计滤波
        p3=cloud.seg_cluster(p2)  #聚类分割        
        p4=remove_hidden_point2(p3)   #添加隐藏点区分目标
        pcd_new = o3d.geometry.PointCloud.uniform_down_sample(p4, 5)  #均匀下采样
        o3d.io.write_point_cloud('E:/linux_data/utile/experimental/res/prepcd/'+img_path[:-4]+'_'+str(ids+1)+'.pcd', pcd_new, True)    


#自定义展示点云
def vis_draw_geometries(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()	#创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()	#设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])	#设置背景色（这里为黑色）
    render_option.point_size = 5	#设置渲染点的大小
    vis.add_geometry(pcd)	#添加点云
    vis.run()

def transform_mask(path):
    img=cv2.imread(path)
    indexs=np.where(img[:,:,-1]!=0)
    img[indexs]=255
    img_show('res',img)
    cv2.imwrite('E:/linux_data/utile/mask01.png',img)


#################################
#预测，中间预处理过程
#################################

def pointProcess(img,mask,depth_path):
    depth_map = depth_path + img[:-4] + '.png'  # 对应深度图路径
    # 获取掩码像素点,或全图像素点
    mask_index = np.where(mask != 0)
    piex = np.array(list(zip(*mask_index)))

    # 读取相机内外参数
    # pose_path=from_prePath_find_posefile(img_path,tar_path,root_path) #每张图片对应相机外参 pose 路径
    # R,T=from_lastPath_find_pose(pose_path,img_path) #相机外参
    C = img_name_to_intrinsic(img)  # 相机原始内参

    # 获取关键点三维信息
    points = convert_2D_to_3D(piex, depth_map, C)

    # 可视化，将点云转换成open3d中的数据形式并用pcd来保存，以方便用open3d处理
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])#按顺序全部赋色
    o3d.visualization.draw_geometries([pcd])

    # 预处理：
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 全部设为灰色
    p2 = cloud.reomove_statistical(pcd)  # 统计滤波
    p3 = cloud.seg_cluster(p2)  # 聚类分割
    p4 = remove_hidden_point2(p3)  # 添加隐藏点区分目标
    p5 = o3d.geometry.PointCloud.uniform_down_sample(p4, 5)  # 均匀下采样
    # 固定数量
    point = np.asarray(p5.points)
    color = np.asarray(p5.colors) * 255
    points_pc = np.concatenate([point, color], axis=1)
    choice = np.random.choice(points_pc.shape[0], 4096, replace=True)
    point_set = points_pc[choice, :]  #[4096,6] （x，y，z，r，g，b）未归一化

    return point_set




if __name__=='__main__':
  
    #transform_mask('E:/linux_data/utile/models/label/cv2_mask/170908_082025654_Camera_5.png')

    '''
    pcd= o3d.io.read_point_cloud("E:/linux_data/utile/experimental/res/pointnetres/171206_062647154_Camera_5_1_0.pcd")   
    #pcd.paint_uniform_color([0.5,0.5,0.5])  
    o3d.visualization.draw_geometries([pcd])
    vis_draw_geometries(pcd) 
    
    
    #generate_pcd()
    #gen_pcd()  
    #kitti_gen_pcd()
    #showProcess()
    
    rootPath='E:/linux_data/utile/experimental/res/img'
    imgList=os.listdir(rootPath)
    for img in imgList:
        showPreProcess(img)
    '''
     
    pcd= o3d.io.read_point_cloud("E:/linux_data/utile/experimental/res/pcd/170908_074956529_Camera_5_1.pcd")   
    o3d.visualization.draw_geometries([pcd])
    vis_draw_geometries(pcd)   
    
    pcd.paint_uniform_color([0.5,0.5,0.5])  # 全部设为灰色
    o3d.visualization.draw_geometries([pcd])

    
    p2=cloud.reomove_statistical(pcd) #统计滤波
    #p3=cloud.seg_cluster(p2)  #聚类分割        
    vis_draw_geometries(p2) 
    p4=remove_hidden_point2(p2)   #添加隐藏点区分目标
    
    pcd_new = o3d.geometry.PointCloud.uniform_down_sample(p4, 10)  #均匀下采样
    o3d.visualization.draw_geometries([pcd_new])
    
    
    o3d.io.write_point_cloud('./tuyang01.pcd', p4, True)


