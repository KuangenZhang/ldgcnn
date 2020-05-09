"""
Evaluate the classification accuracy on the ModelNet40 based on our ldgcnn 
trained feature extractor and classifier. We borrow the evaluation code 
from the DGCNN, and add the code of combining the classifier with the 
feature extractor. 
Reference code: https://github.com/WangYueFt/dgcnn
@author: Kuangen Zhang

"""
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import os
import scipy.misc
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'VisionProcess'))
from PlotClass import PlotClass
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model_cnn', default='ldgcnn', help='Model name: dgcnn [default: dgcnn]')
parser.add_argument('--model_fc', default='ldgcnn_classifier', help='Model name: dgcnn [default: dgcnn]')
parser.add_argument('--batch_size', type=int, default= 16, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--num_feature', type=int, default=3072, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
FLAGS = parser.parse_args()

NAME_MODEL = ''
LOG_DIR = FLAGS.log_dir
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FEATURE = FLAGS.num_feature
GPU_INDEX = FLAGS.gpu
# MODEL_CNN: Model of feature extractor (convolutional layers)
MODEL_CNN = importlib.import_module(FLAGS.model_cnn)
# MODEL_FC: Model of feature extractor (convolutional layers)
MODEL_FC = importlib.import_module(FLAGS.model_fc)
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 
HOSTNAME = socket.gethostname()
#%%
# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

is_training = False
#%%
with tf.device('/gpu:'+str(GPU_INDEX)):
    # Input of the MODEL_CNN is the point cloud and label.
    pointclouds_pl, labels_pl = MODEL_CNN.placeholder_inputs(BATCH_SIZE, NUM_POINT)
    # Input of the MODEL_FC is the global feature and label.
    features, labels_features = MODEL_FC.placeholder_inputs(BATCH_SIZE, NUM_FEATURE)
    is_training_pl = tf.placeholder(tf.bool, shape=())

    _, layers = MODEL_CNN.get_model(pointclouds_pl, is_training_pl)
    pred,_ = MODEL_FC.get_model(features, is_training_pl)
    loss = MODEL_FC.get_loss(pred, labels_pl)
    #%%
with tf.device('/gpu:'+str(GPU_INDEX)):    
    # Add ops to save and restore all the variables.
    variable_names = [v.name for v in tf.global_variables()]
    variables = tf.global_variables()
    # Variables before #43 belong to the feature extractor.
    saver_cnn = tf.train.Saver(variables[0:44])
    # Variables after #43 belong to the classifier.
    saver_fc = tf.train.Saver(variables[44:])
#%%
# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True

ops = {'pointclouds_pl': pointclouds_pl,
       'features': features,
       'labels_pl': labels_pl,
       'labels_features': labels_features,
       'is_training_pl': is_training_pl,
       'pred': pred,
       'loss': loss}

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
#%%
i = 0
Files = TEST_FILES
with tf.Session(config=config) as sess:
    with tf.device('/gpu:'+str(GPU_INDEX)):
        #Restore variables of feature extractor from disk.
        saver_cnn.restore(sess, os.path.join(LOG_DIR, FLAGS.model_cnn+'_'+ 
                                             str(NAME_MODEL)+"model.ckpt"))
        #Restore variables of classifier from disk.
        saver_fc.restore(sess, os.path.join(LOG_DIR, FLAGS.model_fc+'_'+ 
                                             str(NAME_MODEL)+"model.ckpt"))
        log_string("Model restored.")
        error_cnt = 0
        is_training = False
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
        global_feature_vec = np.array([])
        label_vec = np.array([])
        for fn in range(len(Files)):
            log_string('----'+str(fn)+'----')
            current_data, current_label = provider.loadDataFile(Files[fn])
            current_data = current_data[:,0:NUM_POINT,:]
            current_label = np.squeeze(current_label)
            print(current_data.shape)
            
            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE
            print(file_size)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE
                cur_batch_size = end_idx - start_idx
                
                # Aggregating begin
                batch_loss_sum = 0 # sum of losses for the batch
                batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
                batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes  
                feed_dict_cnn = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                             ops['labels_pl']: current_label[start_idx:end_idx],
                             ops['is_training_pl']: is_training}
                # Extract the global_feature from the feature extractor.
                global_feature = np.squeeze(layers['global_feature'].eval(
                    feed_dict=feed_dict_cnn))
                
                # I find that we can increase the accuracy by about 0.2% after 
                # padding zero vectors, but I do not know the reason.
                global_feature = np.concatenate([global_feature, np.zeros((
                global_feature.shape[0], NUM_FEATURE - global_feature.shape[1]))], axis  = -1)
                
                # Input the extracted features and labels to the classifier.
                feed_dict = {ops['features']: global_feature,
                             ops['labels_pl']: current_label[start_idx:end_idx],
                             ops['is_training_pl']: is_training}
                # Calculate the loss and classification scores.
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                          feed_dict=feed_dict)
                batch_pred_sum += pred_val
                batch_pred_val = np.argmax(pred_val, 1)
                for el_idx in range(cur_batch_size):
                    batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
                batch_loss_sum += (loss_val * cur_batch_size)
                # pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
                pred_val = np.argmax(batch_pred_sum, 1)
                # Aggregating end
                
                correct = np.sum(pred_val == current_label[start_idx:end_idx])
                # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
                total_correct += correct
                total_seen += cur_batch_size
                loss_sum += batch_loss_sum
    
                for i in range(start_idx, end_idx):
                    l = current_label[i]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_val[i-start_idx] == l)
                    fout.write('%d, %d\n' % (pred_val[i-start_idx], l))
  
        log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
        log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
        log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
        
        class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
        for i, name in enumerate(SHAPE_NAMES):
            log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
#%%
#calculate confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import re
f = open("dump/pred_label.txt", "r")
str_data = f.read()
data = re.findall(r"[-+]?\d*\.\d+|\d+", str_data)
data = np.array(list(map(int, data)))
data = np.reshape(data, (-1, 2))
f = open("dump/shape_names.txt", "r")
class_names = np.array(f.read().split())
# Plot the confusion matrix
cm,ax = PlotClass.plot_confusion_matrix(data[:,1], data[:,0], classes=class_names, normalize=True,
                      title='Normalized confusion matrix')