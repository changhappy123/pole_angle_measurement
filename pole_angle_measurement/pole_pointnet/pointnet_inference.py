import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from model import *
import indoor3d_util
import h5py
import open3d as o3d
from gen_pole_h5 import *

class Pole_inference():
    def __init__(self):
        self.BATCH_SIZE = 1
        self.NUM_POINT = 4096
        self.MODEL_PATH = os.path.join(ROOT_DIR,'pole_pointnet/log2/model.ckpt')  #模型参数文件
        self.GPU_INDEX = 0
        self.DUMP_DIR = os.path.join(ROOT_DIR,'pole_pointnet/log2/dump')
        if not os.path.exists(self.DUMP_DIR): os.mkdir(self.DUMP_DIR)
        self.LOG_FOUT = open(os.path.join(self.DUMP_DIR, 'log_evaluate.txt'), 'w')
        # self.LOG_FOUT.write(str(FLAGS)+'\n')
        self.NUM_CLASSES = 2
        self.calculation = False
        self.visu = True
        self.no_clutter= False

        # #测试文件
        # inpath='pole_pointnet/test_points'
        # inpath=os.path.join(ROOT_DIR,inpath)
        # ROOM_PATH_LIST =[os.path.join(inpath,pointpath) for pointpath in os.listdir('test_points')]


    def log_string(self,out_str):
        self.LOG_FOUT.write(out_str+'\n')
        self.LOG_FOUT.flush()
        print(out_str)

    def evaluate(self,npdata=None):
        is_training = False
        with tf.device('/gpu:'+str(self.GPU_INDEX)):

            pointclouds_pl, labels_pl = placeholder_inputs(self.BATCH_SIZE, self.NUM_POINT)  #1.输入 placeholder
            is_training_pl = tf.placeholder(tf.bool, shape=())
            # simple model
            pred = get_model(pointclouds_pl, is_training_pl)   #2.获取模型和损失函数
            pred_softmax = tf.nn.softmax(pred)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()  # 3.新建saver 加载参数

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # 动态申请缓存
            config.allow_soft_placement = True   #如果是 True，允许 tensorflow 自动分配设备
            config.log_device_placement = True   #如果是 True，打印设备分配日志
            sess = tf.Session(config=config)  #4.创建会话

            # Restore variables from disk.
            saver.restore(sess, self.MODEL_PATH)
            self.log_string("Model restored.")

            if self.calculation:
                assert self.ROOM_PATH_LIST[0].endswith('.txt'), '文件格式错误'
                # simple loss
                loss = get_loss(pred, labels_pl)

                ops = {'pointclouds_pl': pointclouds_pl,
                       'labels_pl': labels_pl,
                       'is_training_pl': is_training_pl,
                       'pred': pred,
                       'pred_softmax': pred_softmax,
                       'loss': loss}

                total_correct,total_seen = self.eval_one_epoch(sess, ops,self.ROOM_PATH_LIST)
                self.log_string('all room eval accuracy: %f' % (total_correct / float(total_seen)))
            else:

                ops = {'pointclouds_pl': pointclouds_pl,
                       'is_training_pl': is_training_pl,
                       'pred_softmax': pred_softmax,
                       }

                targetpc=self.ModelPredict(sess, ops, npdata)
                return targetpc


    def eval_one_epoch(self,sess, ops, room_path):
        error_cnt = 0
        is_training = False
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(self.NUM_CLASSES)]
        total_correct_class = [0 for _ in range(self.NUM_CLASSES)]

        file_size = len(room_path)
        num_batches = file_size // self.BATCH_SIZE
        print(file_size)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.BATCH_SIZE
            end_idx = (batch_idx+1) * self.BATCH_SIZE
            cur_batch_size = end_idx - start_idx

            # 输入数据和标签
            original_data,current_data, current_label =load_point(room_path[batch_idx])
            current_data= np.expand_dims(current_data,axis=0)    #[1,4096,6]
            current_label= np.expand_dims(current_label,axis=0)  #[1,4096]


            feed_dict = {ops['pointclouds_pl']: current_data,
                         ops['labels_pl']: current_label,
                         ops['is_training_pl']: is_training}    # 5. feed_dict 传参数
            loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],   #6. 运行会话
                                          feed_dict=feed_dict)    #pred_val [1,4096,2]

            if self.no_clutter:
                pred_label = np.argmax(pred_val[:,:,0:12], 2) # BxN
            else:
                pred_label = np.argmax(pred_val, 2) # BxN    [1,4096]

            if self.visu:
                self.vis(original_data)
                self.vis(original_data, pred_label, predict=True)

            correct = np.sum(pred_label == current_label)  #每4096个点中正确的点数
            total_correct += correct  #总正确的点数
            total_seen += (cur_batch_size* self.NUM_POINT)  #总个数
            loss_sum += (loss_val* self.BATCH_SIZE)  #总损失值

            for j in range(self.NUM_POINT):
                l = current_label[0, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_label[0, j] == l)

        self.log_string('eval mean loss: %f' % (loss_sum / float(total_seen/self.NUM_POINT)))
        self.log_string('eval accuracy: %f'% (total_correct / float(total_seen)))

        return total_correct, total_seen


    #预测函数,返回电杆分割点云
    def ModelPredict(self, sess, ops, npdata):   #npdata [4096,6] 为2D像素转换为3D数据，经过统计滤波、聚类分割、隐藏点和下采样及数量固定
        # 输入数据和标签
        is_training = False
        current_data = load_npdata_point(npdata)
        current_data = np.expand_dims(current_data, axis=0)

        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['is_training_pl']: is_training}

        pred_val = sess.run(ops['pred_softmax'],
                              feed_dict=feed_dict)

        if self.no_clutter:
            pred_label = np.argmax(pred_val[:, :, 0:12], 2)  # BxN
        else:
            pred_label = np.argmax(pred_val, 2)  # BxN

        ids=np.where(pred_label[0]==1)[0]
        npdata[:, 3:6] = npdata[:, 3:6] / 255
        targetpc = npdata[ids]

        if self.visu:
            self.vis(npdata)
            self.vis(npdata, pred_label, predict=True)
            self.vis(targetpc)
        return targetpc


    #展示预测结果
    def vis(self, pc,pred_label=None,predict=False):    #pc [4096,6] pre_label [1,4096]
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(pc[:,0:3])
        pcd.colors=o3d.utility.Vector3dVector(pc[:,3:6])

        # 点云显示
        if predict:
            pcd.paint_uniform_color([0.5,0.5,0.5])
            pre_label=np.squeeze(pred_label)
            idx = np.where(pre_label==1)[0]
            for i in list(idx):
                pcd.colors[i]=[1,0,0]

            o3d.visualization.draw_geometries([pcd],  # 待显示的点云列表
                                              window_name="预测点云显示",
                                              point_show_normal=False,
                                              width=800,  # 窗口宽度
                                              height=600)  # 窗口高度

            #o3d.io.write_point_cloud('/home/chang/my/Master/pole_angle_detection/showres/' + '00.pcd', pcd, True)

        else:
            o3d.visualization.draw_geometries([pcd],  # 待显示的点云列表
                                              window_name="原始点云显示",
                                              point_show_normal=False,
                                              width=800,  # 窗口宽度
                                              height=600)  # 窗口高度

    def run(self, original_data):
        with tf.Graph().as_default():
            targetpc=self.evaluate(original_data)
        self.LOG_FOUT.close()
        return targetpc


if __name__=='__main__':
    pc=o3d.io.read_point_cloud('test2/170927_074138590_Camera_6_1.pcd')
    point=np.asarray(pc.points)
    color=np.asarray(pc.colors)*255
    points=np.concatenate([point,color],axis=1)

    # points=np.loadtxt('test_points/170908_082025062_Camera_5_1.txt')
    # points=points[:,0:6]

    choice = np.random.choice(points.shape[0], 4096, replace=True)
    point_set = points[choice, :]
    # original data
    original_data = point_set[:, 0:6]

    dete=Pole_inference()
    with tf.Graph().as_default():
        dete.evaluate(original_data)
    dete.LOG_FOUT.close()

    # with tf.Graph().as_default():
    #     evaluate()
    # LOG_FOUT.close()