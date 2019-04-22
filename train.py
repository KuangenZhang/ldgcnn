"""
Training process of linked dynamic graph CNN. We borrow the training code 
from the DGCNN, and add the code of retraining classifier. 
Reference code: https://github.com/WangYueFt/dgcnn
@author: Kuangen Zhang

"""
import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'VisionProcess'))
import provider
from FileIO import FileIO

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='ldgcnn', help='Model name: dgcnn')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default= 32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')

# The parameters of retrained classifier
parser.add_argument('--model_classifier', default='ldgcnn_classifier', help='Model name: dgcnn')
parser.add_argument('--num_feature_classifier', type=int, default= 3072, help='Point Number [1024/1984] [default: 1024]')
parser.add_argument('--max_epoch_classifier', type=int, default=100, help='Epoch to run [default: 250]')
parser.add_argument('--optimizer_classifier', default='momentum', help='adam or momentum [default: adam]')

FLAGS = parser.parse_args()

NUM_FEATURE_CLS = FLAGS.num_feature_classifier
OPTIMIZER_CLS = FLAGS.optimizer_classifier
MAX_EPOCH_CLS = FLAGS.max_epoch_classifier
MODEL_CLS = importlib.import_module(FLAGS.model_classifier) # import network module
MODEL_FILE_CLS = os.path.join(BASE_DIR, 'models', FLAGS.model_classifier+'.py')

NAME_MODEL = ''
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (MODEL_FILE_CLS, LOG_DIR)) # bkp of model_cls def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR,FLAGS.model + NAME_MODEL + '_log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
folder = 'data/modelnet40_ply_hdf5_2048'
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, folder + '/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, folder + '/test_files.txt'))

# Feature files, which are generated after training the whole network.
# The extracted feature files are utilized to train the classifier.
path = 'data/extracted_feature'
TRAIN_FILES_CLS = provider.getDataFiles( \
    os.path.join(BASE_DIR, path + '/train_files.txt'))
TEST_FILES_CLS = provider.getDataFiles(\
    os.path.join(BASE_DIR, path + '/test_files.txt'))

# Print the log contents to a txt file.
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Decay the learning rate to avoid oscillation.
def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step = batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' 
            # parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred,layers = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)
            
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training optimizer
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(
                LOG_DIR, FLAGS.model + NAME_MODEL +'_train'),sess.graph) 
        test_writer = tf.summary.FileWriter(os.path.join(
                LOG_DIR,FLAGS.model + NAME_MODEL +'_test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

#        # restore paramters
#        saver.restore(sess, os.path.join(LOG_DIR, FLAGS.model+
#                                             str(NAME_MODEL)+"_model.ckpt"))
        
        best_accuracy = 0.0
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            train_one_epoch(sess, ops, train_writer)
            accuracy = eval_one_epoch(sess, ops, test_writer)
            # Save the network that achieves the best validation accuracy.
            # There are only training set and validation set for ModelNet40. 
            # Previous researchers report their best accuracy rather than 
            # final accuracy because they also regard the testing set as 
            # validation set.
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_path = saver.save(sess, os.path.join(
                        LOG_DIR, FLAGS.model+ NAME_MODEL +"_model.ckpt"))
                log_string("Best accuracy, model saved in file: %s" % save_path)
        
        # Save the extracted global feature 
        save_global_feature(sess, ops, saver,layers)
                

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        # Load data and labels from the files.
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        # Shuffle the data in the training set.
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
       
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotating, jittering, shifting, 
            # and scaling.
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            jittered_data = provider.random_scale_point_cloud(jittered_data)
            jittered_data = provider.rotate_perturbation_point_cloud(jittered_data)
            jittered_data = provider.shift_point_cloud(jittered_data)
            
            # Input the augmented point cloud and labels to the graph.
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            
            # Calculate the loss and accuracy of the input batch data.            
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            # Input the point cloud and labels to the graph.
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            # Calculate the loss and classification scores.
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    return total_correct / float(total_seen)  

def save_global_feature(sess, ops, saver, layers):
    feature_name = 'global_feature'
    file_name_vec = ['train_' + feature_name, 'test_' + feature_name]
    Files_vec = [TRAIN_FILES, TEST_FILES]
    #Restore variables that achieves the best validation accuracy from the disk.
    saver.restore(sess, os.path.join(LOG_DIR, FLAGS.model+
                                         str(NAME_MODEL)+ "_model.ckpt")) 
    log_string("Model restored.") 
    is_training = False
    # Extract the features from training set and validation set.
    for r in range(2):
        file_name = file_name_vec[r]
        Files = Files_vec[r]
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
                # Input the point cloud and labels to the graph.
                feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                             ops['labels_pl']: current_label[start_idx:end_idx],
                             ops['is_training_pl']: is_training}
                # Extract the global features from the input batch data.
                global_feature = np.squeeze(layers[feature_name].eval(
                    feed_dict=feed_dict,session=sess))
                
                if label_vec.shape[0] == 0:
                    global_feature_vec = global_feature
                    label_vec = current_label[start_idx:end_idx]
                else:
                    global_feature_vec = np.concatenate([global_feature_vec, global_feature])
                    label_vec = np.concatenate([label_vec, current_label[start_idx:end_idx]])      
        # Save all global features to the disk.
        FileIO.write_h5('data/extracted_feature/' + file_name + '.h5', global_feature_vec, label_vec)

def train_classifier():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL_CLS.placeholder_inputs(
                    BATCH_SIZE, NUM_FEATURE_CLS)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, layers = MODEL_CLS.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            
            loss = MODEL_CLS.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            # We change the optimier to momentum. Because the momentum can 
            # find better parameters
            if OPTIMIZER_CLS == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER_CLS == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif OPTIMIZER_CLS == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train_'+FLAGS.model+'_'+str(NAME_MODEL)),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test_'+FLAGS.model+'_'+str(NAME_MODEL)),
                                  sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        
        # We retrain the classifier three times to see the stable accuracy.
        # It takes much less time to retrain the classifier than to 
        # train the whole network. 
        idx_num = 3
        best_accuracy_vec = np.zeros(idx_num)
        class_accuracy_vec = np.zeros(idx_num)
        save_path = ''
        for idx in range(idx_num):
            best_accuracy = 0.0
            # Init variables
            init = tf.global_variables_initializer()
            # To fix the bug introduced in TF 0.12.1 as in
            # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
            #sess.run(init)
            sess.run(init, {is_training_pl: True})
    
            ops = {'pointclouds_pl': pointclouds_pl,
                   'labels_pl': labels_pl,
                   'is_training_pl': is_training_pl,
                   'pred': pred,
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch}
            
            for epoch in range(MAX_EPOCH_CLS):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()
                
                train_classifier_one_epoch(sess, ops, train_writer)
                accuracy, class_accuracy = eval_classifier_one_epoch(sess, ops, test_writer)
    
                # Save the variables that achieves the best accuracy to disk.
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_accuracy_vec[idx:] = accuracy
                    class_accuracy_vec[idx:] = class_accuracy
                    log_string('Best accuracy,: %f'% best_accuracy)
                    log_string('Class accuracy,: %f'% class_accuracy)
                    save_path = saver.save(sess, os.path.join(LOG_DIR, 
                                                FLAGS.model_classifier+str(NAME_MODEL)+"_model.ckpt"))
        
        log_string('Max of best accuracy,: %f'% np.max(best_accuracy_vec))
        print(best_accuracy_vec)
        log_string('Max of class accuracy,: %f'% np.max(class_accuracy_vec))
        log_string("Model saved in file: %s" % save_path)
        print(class_accuracy_vec) 
                

def train_classifier_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    for fn in range(len(TRAIN_FILES_CLS)):    
        # Shuffle train files
        current_data, current_label = provider.loadDataFile(TRAIN_FILES_CLS[fn]) 
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)
        # I find that we can increase the accuracy by about 0.2% after 
        # padding zero vectors, but I do not know the reason.
        current_data = np.concatenate([current_data, np.zeros((
                current_data.shape[0], NUM_FEATURE_CLS - current_data.shape[1]))], axis  = -1)
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Input the features and labels to the graph.
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx,...],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            # Calculate the loss and classification scores.
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
                    
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val

def eval_classifier_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for  _ in range(NUM_CLASSES)]
    file_size_sum = 0
    for fn in range(len(TEST_FILES_CLS)):
        current_data, current_label = provider.loadDataFile(TEST_FILES_CLS[fn])
        current_label = np.squeeze(current_label)
        # I find that we can increase the accuracy by about 0.2% after 
        # padding zero vectors, but I do not know the reason.
        current_data = np.concatenate([current_data, np.zeros((
                current_data.shape[0], NUM_FEATURE_CLS - current_data.shape[1]))], axis  = -1)
        
        file_size = current_data.shape[0]
        file_size_sum += file_size
        num_batches = file_size // BATCH_SIZE
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            # Input the features and labels to the graph.
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx,:],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            # Calculate the loss and classification scores.
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            test_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
    accuracy = total_correct / float(total_seen)
    class_accuracy = np.mean(np.array(total_correct_class)/np.array(
            total_seen_class,dtype=np.float))
    return accuracy, class_accuracy  
    

if __name__ == "__main__":
    train()
    train_classifier()
    LOG_FOUT.close()