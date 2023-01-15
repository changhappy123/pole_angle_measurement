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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--dump_dir', required=True, help='dump folder path')
parser.add_argument('--output_filelist', required=True, help='TXT filename, filelist, each line is an output for a room')
parser.add_argument('--room_data_filelist', required=True, help='TXT filename, filelist, each line is a test_points room data label file.')
parser.add_argument('--no_clutter', action='store_true', help='If true, donot count the clutter class')
parser.add_argument('--visu', action='store_true', help='Whether to output OBJ file for prediction visualization.')
parser.add_argument('--calculation', default=False, help='Whether the correct rate is calculated')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
#ROOM_PATH_LIST = [os.path.join(ROOT_DIR,line.rstrip()) for line in open(FLAGS.room_data_filelist)]
NUM_CLASSES = 2
calculation = FLAGS.calculation

#测试文件
inpath='pole_pointnet/test_points'
inpath=os.path.join(ROOT_DIR,inpath)
ROOM_PATH_LIST =[os.path.join(inpath,pointpath) for pointpath in os.listdir('test_points')]


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(npdata=None):
    is_training = False
    with tf.device('/gpu:'+str(GPU_INDEX)):

        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)  #1.输入 placeholder
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
        saver.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        if calculation:
            assert ROOM_PATH_LIST[0].endswith('.txt'), '文件格式错误'
            # simple loss
            loss = get_loss(pred, labels_pl)

            ops = {'pointclouds_pl': pointclouds_pl,
                   'labels_pl': labels_pl,
                   'is_training_pl': is_training_pl,
                   'pred': pred,
                   'pred_softmax': pred_softmax,
                   'loss': loss}

            total_correct,total_seen = eval_one_epoch(sess, ops,ROOM_PATH_LIST)
            log_string('all room eval accuracy: %f' % (total_correct / float(total_seen)))
        else:

            ops = {'pointclouds_pl': pointclouds_pl,
                   'is_training_pl': is_training_pl,
                   'pred_softmax': pred_softmax,
                   }

            targetpc=ModelPredict(sess, ops, npdata)
            return targetpc


def eval_one_epoch(sess, ops, room_path):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    file_size = len(room_path)
    num_batches = file_size // BATCH_SIZE
    print(file_size)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
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

        if FLAGS.no_clutter:
            pred_label = np.argmax(pred_val[:,:,0:12], 2) # BxN
        else:
            pred_label = np.argmax(pred_val, 2) # BxN    [1,4096]

        if FLAGS.visu:
            vis(original_data)
            vis(original_data, pred_label, predict=True)

        correct = np.sum(pred_label == current_label)  #每4096个点中正确的点数
        total_correct += correct  #总正确的点数
        total_seen += (cur_batch_size*NUM_POINT)  #总个数
        loss_sum += (loss_val*BATCH_SIZE)  #总损失值

        for j in range(NUM_POINT):
            l = current_label[0, j]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_label[0, j] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))

    return total_correct, total_seen


#预测函数,返回电杆分割点云
def ModelPredict(sess, ops, npdata):   #npdata [4096,6] 为2D像素转换为3D数据，经过统计滤波、聚类分割、隐藏点和下采样
    # 输入数据和标签
    is_training = False
    current_data = load_npdata_point(npdata)
    current_data = np.expand_dims(current_data, axis=0)

    feed_dict = {ops['pointclouds_pl']: current_data,
                 ops['is_training_pl']: is_training}

    pred_val = sess.run(ops['pred_softmax'],
                          feed_dict=feed_dict)

    if FLAGS.no_clutter:
        pred_label = np.argmax(pred_val[:, :, 0:12], 2)  # BxN
    else:
        pred_label = np.argmax(pred_val, 2)  # BxN

    ids=np.where(pred_label[0]==1)[0]
    npdata[:, 3:6] = npdata[:, 3:6] / 255
    targetpc = npdata[ids]

    if FLAGS.visu:
        vis(npdata)
        vis(npdata, pred_label, predict=True)
        vis(targetpc)
    return targetpc


#展示预测结果
def vis(pc,pred_label=None,predict=False):    #pc [4096,6] pre_label [1,4096]
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
    else:
        o3d.visualization.draw_geometries([pcd],  # 待显示的点云列表
                                          window_name="原始点云显示",
                                          point_show_normal=False,
                                          width=800,  # 窗口宽度
                                          height=600)  # 窗口高度


if __name__=='__main__':
    '''
    pc=o3d.io.read_point_cloud('test2/170908_082024914_Camera_5_1.pcd')
    point=np.asarray(pc.points)
    color=np.asarray(pc.colors)*255
    points=np.concatenate([point,color],axis=1)

    # points=np.loadtxt('test_points/170908_082025062_Camera_5_1.txt')
    # points=points[:,0:6]

    choice = np.random.choice(points.shape[0], 4096, replace=True)
    point_set = points[choice, :]
    # original data
    original_data = point_set[:, 0:6]

    with tf.Graph().as_default():
        evaluate(original_data)
    LOG_FOUT.close()
   '''

    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()