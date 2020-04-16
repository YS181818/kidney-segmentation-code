## Training the Bnet
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
import scipy.io as sio
import scipy.misc as misc
import scipy.ndimage as ndi
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
TEST_LIST_PATH = '../../../tensorflow-deeplab-lfov-master/kid/test_80.txt'
TRAIN_LIST_PATH = '../../../tensorflow-deeplab-lfov-master/kid/kid_warp/Image_warp/train_warp.txt'
IMAGE_PATH = '../../../tensorflow-deeplab-lfov-master/kid/kid_warp/'
LABEL_PATH = '../../../tensorflow-deeplab-lfov-master/kid/kid_warp/'
WEIGHTS_PATH   = '../../../tensorflow-deeplab-lfov-master/util/VGG_16.npy'
INPUT_SIZE = (321,321)
Base_Rate = 1e-5
NUM_STEPS = 20001
batch_size = 20
SAVE_NUM_IMAGES = 10
snapshot_dir = './result/aug_3VGG80/snapshots'

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
    

    


    
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
    net_deeplab = DeepLabLFOVModel(WEIGHTS_PATH)     
    net_regression = RegressionNet()
    _,_,_,train_feature1024 = net_deeplab.preds(trainimage_batch)

    
    # test
    _,_,_,test_feature1024 = net_deeplab.preds(testimage_batch)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    trainloss_old = tf.placeholder(dtype=tf.float32, shape=())
    # compute loss and show
    with tf.name_scope('train'):
        train_reg_loss = net_regression.getloss(train_feature1024, trainlabel_distance321,batch_size)
        trainpred_regression = net_regression.preds(train_feature1024,batch_size)
        trainpred_regression321 = trainpred_regression
        trainloss = train_reg_loss
        trainloss_summary = tf.summary.scalar('trainloss', trainloss)
        train_image_summary1 = tf.summary.image('train_pred1',
                                  tf.concat([
                                            trainimage_batch,
                                            tf.concat([trainmask_batch*255,trainmask_batch*255,trainmask_batch*255],axis=3),
                                            tf.concat([trainlabel_distance321*255,trainlabel_distance321*255,trainlabel_distance321*255],axis=3),
                                            tf.concat([trainpred_regression321*255,trainpred_regression321*255,trainpred_regression321*255],axis=3),                                                                                        
                                          ],axis=2),
                                 SAVE_NUM_IMAGES) 



                               
    with tf.name_scope('test'):
        test_reg_loss = net_regression.getloss(test_feature1024, testlabel_distance321,batch_size)
        testpred_regression = net_regression.preds(test_feature1024,batch_size)
        testpred_regression321 = testpred_regression
        testloss = test_reg_loss
        testloss_summary = tf.summary.scalar('testloss', testloss)
        test_image_summary1 = tf.summary.image('test_pred1',
                                  tf.concat([
                                            testimage_batch,
                                            tf.concat([testmask_batch*255,testmask_batch*255,testmask_batch*255],axis=3), 
                                            tf.concat([testlabel_distance321*255,testlabel_distance321*255,testlabel_distance321*255],axis=3),
                                            tf.concat([testpred_regression321*255,testpred_regression321*255,testpred_regression321*255],axis=3),                                                                                                
                                          ],axis=2),
                                 SAVE_NUM_IMAGES) 

    trainloss_sum= trainloss_old+trainloss                               
    Learning_Rate = tf.scalar_mul(Base_Rate, tf.pow((1 - step_ph/NUM_STEPS), 0.5))
    optimiser = tf.train.AdamOptimizer(learning_rate=Learning_Rate)
    trainable = tf.trainable_variables()
    optim = optimiser.minimize(trainloss_sum, var_list=trainable)
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    saver = tf.train.Saver(var_list=trainable, max_to_keep=20)
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    file_writer = tf.summary.FileWriter('./result/log', sess.graph)
    train_writer = tf.summary.FileWriter('./result/log' + '/train')
    test_writer = tf.summary.FileWriter('./result/log' + '/test')
    tmp_ = [train_image_summary1,trainloss_summary]
    merged_train = tf.summary.merge(tmp_)
    tmp_ = [test_image_summary1,testloss_summary]
    merged_test = tf.summary.merge(tmp_)
    for step in range(NUM_STEPS):
        start_time = time.time()
        feed_dict0 = { step_ph: step, trainloss_old: 0}
        _, loss_train, trainsummary, loss_train_reg = sess.run([optim, trainloss_sum, merged_train, train_reg_loss], feed_dict=feed_dict0)
        train_writer.add_summary(trainsummary, step)
        duration = time.time() - start_time
        if step % 10 == 0:
              print('step {:d} \t trainloss = {:.3f}, ({:.3f} sec/step),traindicemask = {:.3f}'.format(step, loss_train, duration))
        if step % 50 == 0:
              testsummary, loss_test, loss_test_reg = sess.run([merged_test, testloss, test_reg_loss], feed_dict=feed_dict0)
              print('step {:d} \t testloss = {:.3f}, ({:.3f} sec/step),testdicemask = {:.3f}'.format(step, loss_test, duration))
              test_writer.add_summary(testsummary, step)
        if step % 1000== 0:
              save(saver, sess, snapshot_dir, step)
               
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()