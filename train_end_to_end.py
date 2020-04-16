
import os
import numpy as np
import tensorflow as tf
import notebook.nbextensions
import zipfile
from six.moves.urllib.request import urlretrieve 
import argparse
from datetime import datetime
import sys
import time
import matplotlib.pyplot as plt
from six.moves import cPickle
from reader_aug_mask import Image_Reader
from regressionnet2_upsmapling import RegressionNet
from deeplab_lfov import DeepLabLFOVModel 
from deeplab_lfov import DeepLabSEGModel ##transfer learning for pixel-wise classification network
import scipy.io as sio
import scipy.misc as misc
import scipy.ndimage as ndi
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]  ## choise GPU 
TEST_LIST_PATH = '../../../tensorflow-deeplab-lfov-master/kid/kid_warp/Image_warp/test_warp.txt'
TRAIN_LIST_PATH = '../../../tensorflow-deeplab-lfov-master/kid/kid_warp/Image_warp/train_warp.txt'
IMAGE_PATH = '../../../tensorflow-deeplab-lfov-master/kid/kid_warp/'
LABEL_PATH = '../../../tensorflow-deeplab-lfov-master/kid/kid_warp/'
WEIGHTS_PATH   = '../../../tensorflow-deeplab-lfov-master/util/VGG_16.npy'
INPUT_SIZE = (321,321)
Base_Rate = 1e-5
NUM_STEPS = 20001
batch_size = 5
SAVE_NUM_IMAGES = 10 

snapshot_dir = './result/aug_VGG_multilossdeeplabmask_1/snapshots' ## snapshot_dir
logdir = './result/aug_VGG_multilossdeeplabmask_1/' ## logdir


if not os.path.exists(logdir):
    os.makedirs(logdir)


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
    

## compute dice index     
def dice_mask(input_batch1, input_batch2):
    input_1=tf.equal(input_batch1, 1.0, name=None)
    print ("input_1:"+repr(input_1))
    input_2=tf.equal(input_batch2, 1.0, name= None)
    print ("input_2:"+repr(input_2))
    accuracy_n=tf.logical_and(input_1, input_2,name=None)
    input2_z=tf.cast(input_2, tf.int32)
    input1_z=tf.cast(input_1, tf.int32)
    accuracy_z=tf.cast(accuracy_n, tf.int32)
    accuracy_n= 2*tf.reduce_sum(accuracy_z)/(tf.reduce_sum(input1_z)+tf.reduce_sum(input2_z))
    print ("accuracy_mask:"+repr(accuracy_n))
    return accuracy_n

    
def main():
    # read train and test pred and label images
    coord = tf.train.Coordinator()
    trainreader = Image_Reader(
                 IMAGE_PATH,
                 LABEL_PATH,
           TRAIN_LIST_PATH,
              INPUT_SIZE,
              )
    trainimage_batch, trainmask_batch, trainlabel_distance321 = trainreader.dequeue(batch_size)


    testreader = Image_Reader(
                     IMAGE_PATH,
                     LABEL_PATH,
                 TEST_LIST_PATH,
                 INPUT_SIZE,
                 )
    testimage_batch, testmask_batch, testlabel_distance321 = testreader.dequeue(batch_size)

    # bulid network
    net_deeplab = DeepLabLFOVModel(WEIGHTS_PATH)   ##transfer learning for regression network  
    net_regression = RegressionNet()## boundary distance regression network
    net_seg = DeepLabSEGModel(WEIGHTS_PATH)## pixelwise classification network
    ## input train dataset
    _,_,_,train_feature1024 = net_deeplab.preds(trainimage_batch)
    trainpred_regression = net_regression.preds(train_feature1024,batch_size)
    trainpred_input = tf.concat([trainpred_regression*255, trainpred_regression*255, trainpred_regression*255], 3)
    train_seg_mask,_,_,_ = net_seg.preds(trainpred_input) 
    # input validation dataset
    test_deeplab_mask,_,_,test_feature1024 = net_deeplab.preds(testimage_batch)
    testpred_regression = net_regression.preds(test_feature1024,batch_size)
    testpred_input = tf.concat([testpred_regression*255, testpred_regression*255, testpred_regression*255], 3)
    test_seg_mask,_,_,_ = net_seg.preds(testpred_input)     
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    trainloss_old = tf.placeholder(dtype=tf.float32, shape=())
    # compute loss and show
    with tf.name_scope('train'):
        train_reg_loss = net_regression.getloss(train_feature1024, trainlabel_distance321,batch_size)        
        trainmask_reglabel = tf.py_func(pred_to_mask,[trainlabel_distance321],tf.float32)
        trainmask_seglabel = trainmask_batch
        trainmask_regpred = tf.py_func(pred_to_mask,[trainpred_regression],tf.float32)
        trainmask_segpred = train_seg_mask 
        train_seg_loss = net_seg.loss(trainpred_input, tf.cast(trainmask_seglabel,tf.uint8))
        traindice_seg = dice_mask(tf.cast(trainmask_segpred,tf.float32), trainmask_seglabel)
        traindice_reg = dice_mask(trainmask_regpred, trainmask_reglabel)
        trainloss = (1-1.0*step_ph/NUM_STEPS)*train_reg_loss+(1.0*step_ph/NUM_STEPS)*train_seg_loss ## multiloss function

                               
    with tf.name_scope('test'):
        test_reg_loss = net_regression.getloss(test_feature1024, testlabel_distance321,batch_size)        
        testmask_reglabel = tf.py_func(pred_to_mask,[testlabel_distance321],tf.float32)
        testmask_seglabel = testmask_batch
        testmask_regpred = tf.py_func(pred_to_mask,[testpred_regression],tf.float32)
        testmask_segpred = test_seg_mask 
        test_seg_loss = net_seg.loss(testpred_input, tf.cast(testmask_seglabel,tf.uint8))
        testdice_seg = dice_mask(tf.cast(testmask_segpred,tf.float32), testmask_seglabel)
        testdice_reg = dice_mask(testmask_regpred, testmask_reglabel)
        testloss = (1-1.0*step_ph/NUM_STEPS)*test_reg_loss+(1.0*step_ph/NUM_STEPS)*test_seg_loss ##multiloss function

         

    trainloss_sum= trainloss_old+trainloss                               
    Learning_Rate = tf.scalar_mul(Base_Rate, tf.pow((1 - step_ph/NUM_STEPS), 0.5))
    optimiser = tf.train.AdamOptimizer(learning_rate=Learning_Rate)
    trainable = tf.trainable_variables()
    print("trainable"+repr(trainable))
    optim = optimiser.minimize(trainloss_sum, var_list=trainable)
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    saver = tf.train.Saver(var_list=trainable, max_to_keep=20)
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)


    for step in range(NUM_STEPS):
        start_time = time.time()
        feed_dict0 = { step_ph: step, trainloss_old: 0}
        oldloss_train = 0.0
        oldreg_losstrain=0.0
        oldseg_losstrain=0.0
        olddice_segtrain = 0.0 
        for cstep in range(3):
            feed_dict0 = { step_ph: step, trainloss_old: 0}
            loss_train,  loss_train_reg, loss_train_seg, diceseg_train = sess.run([trainloss, train_reg_loss, train_seg_loss, traindice_seg], feed_dict=feed_dict0)   
            oldloss_train = oldloss_train+loss_train
            oldreg_losstrain = oldreg_losstrain+loss_train_reg
            oldseg_losstrain = oldseg_losstrain+loss_train_seg
            olddice_segtrain = olddice_segtrain + diceseg_train           
        feed_dict1 = { step_ph: step, trainloss_old: oldloss_train}
        _, loss_trainsum, loss_train_reg, loss_train_seg, diceseg_train = sess.run([optim, trainloss_sum, train_reg_loss, train_seg_loss, traindice_seg], feed_dict=feed_dict1)
        oldreg_losstrain = oldreg_losstrain+loss_train_reg
        oldseg_losstrain = oldseg_losstrain+loss_train_seg
        olddice_segtrain = olddice_segtrain + diceseg_train 
        duration = time.time() - start_time
        if step % 10 == 0:
              print('step {:d} \t ({:.3f} sec/step), trainloss = {:.3f},train_regloss = {:.3f},train_segloss = {:.3f}, train_segdice = = {:.3f}'.format(step, duration, loss_trainsum, oldreg_losstrain, oldseg_losstrain, olddice_regtrain/4.0, olddice_segtrain/4.0))
        if step % 50 == 0:
              oldloss_test = 0.0
              oldreg_losstest=0.0
              oldseg_losstest=0.0
              olddice_segtest = 0.0 
              for cstep in range(4):
                  loss_test, loss_test_reg, loss_test_seg,diceseg_test = sess.run([testloss, test_reg_loss, test_seg_loss, testdice_seg], feed_dict=feed_dict0)
                  oldloss_test = oldloss_test+loss_test
                  oldreg_losstest = oldreg_losstest+loss_test_reg
                  oldseg_losstest = oldseg_losstest+loss_test_seg
                  olddice_segtest = olddice_segtest + diceseg_test 
              print('step {:d} \t ({:.3f} sec/step), testloss = {:.3f},test_regloss = {:.3f},test_segloss = {:.3f},test_regdice = {:.3f}, test_segdice = = {:.3f}'.format(step, duration, oldloss_test, oldreg_losstest, oldseg_losstest, olddice_regtest/4.0, olddice_segtest/4.0))
        if step % 1000== 0:
              save(saver, sess, snapshot_dir, step)
               
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()