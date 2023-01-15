# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:49:23 2022

@author: Administrator
"""
#生成电杆点云 h5文件

from sklearn import preprocessing
import numpy as np
import os
import h5py


def load_point(path):
    #加载点云数据集
    '''
    data: [4096,6] (x,y,z,r,g,b)
    label: [4096]
    '''
    point=np.loadtxt(path)
    point_num=4096
    # Resample  重新采样随机选择4096个点云，可以重复
    choice = np.random.choice(point.shape[0], point_num, replace=True)
    point_set = point[choice, :]
    # original data
    original_data=point_set[:,0:6]
    original_data[:,3:6]=original_data[:,3:6]/255
    #train data
    train_data=point_set[:,0:3]
    #标准化
    train_data=preprocessing.StandardScaler().fit_transform(train_data)
    #train_data=(train_data-np.mean(train_data))/np.std(train_data)
    rgb=point_set[:,3:6]/255.0
    data=np.concatenate([train_data,rgb],axis=1)
    
    #train label
    train_label=point_set[:,-1]
    label=train_label.astype(int)
    return original_data,data,label


def load_npdata_point(point_set):
    # train data
    train_data = point_set[:, 0:3]
    # 进行标准化
    train_data = preprocessing.StandardScaler().fit_transform(train_data)
    # train_data=(train_data-np.mean(train_data))/np.std(train_data)
    rgb = point_set[:, 3:6] / 255.0
    data = np.concatenate([train_data, rgb], axis=1)
    return data


#保存为h5文件
def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='int'):
    h5_fout = h5py.File(h5_filename,'a')
    
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=4,
            dtype=label_dtype)
    
    h5_fout.close()


#加载所有文件
def loadfile(path,suffix):
    global file_path
    files=os.listdir(path)
    for file in files:
        if not os.path.isdir(path+file): #判断是不是文件
            if file.endswith(suffix):  #判断是不是txt文件
                file_path.append(path+file)
        else:
            loadfile(path+file+'/',suffix)
    
#生成
def gen_h5(files,h5_filename):
    data=[]
    label=[]
    for file in files:
        train_data,train_label=load_point(file)
        print(file,train_data.shape)
        data.append(train_data)
        label.append(train_label)
    
    data=np.stack(data,axis=0)
    label=np.stack(label,axis=0)
    
    save_h5(h5_filename,data,label)    


if __name__=='__main__':
    
    load_path='E:/linux_data/utile/models/dataset/'
    suffix='.txt'
    file_path=[]
    loadfile(load_path,suffix)
    
    h5_filename='./pole_data.h5'
    gen_h5(file_path,h5_filename)

    #读取数据集
    with h5py.File("models/pole_data.h5","r") as f:
        datas=f['data'][:]
        labels=f['label'][:]
    print(datas.shape)
    print(labels.shape)  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
   