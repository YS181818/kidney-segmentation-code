## Evaluating the end-to-end learning of subsequent segmentation networks
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
from deeplab_lfov import DeepLabSEGModel
import scipy.io as io
import scipy.misc as misc
import scipy.ndimage as ndi
import math
from foo import bwmorph_thin
import networkx as nx
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
TEST_LIST_PATH = 'test.txt'## test image list
IMAGE_PATH = 'kid_image_path/'## image path 
LABEL_PATH = 'kid_label_path/'## label path
WEIGHTS_PATH   = 'VGG_16.npy'## pretrained model
model_weights = 'model_weights/'## the model weights for evaluation
batch_size = 1##the number of test images 
INPUT_SIZE =(321,321)
save_pred_dir = './save_image_path/'## save output images
thred = np.exp(-1.0)

dice_B0_matrix = np.zeros((batch_size,1))
dice_B_matrix = np.zeros((batch_size,1))
dice_M0_matrix = np.zeros((batch_size,1))
dice_M_matrix = np.zeros((batch_size,1))

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask 
    

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


    

def image_slice(image_batchs, index):
    image_s = np.squeeze(image_batchs[index,:,:,:], axis=0)
    return image_s

## read image and label list
def read_pred_label_list(image_dir,label_dir,data_list):
    f = open(data_list, 'r')
    images = []
    preds = []
    masks = []
    pred_regname = []
    for line in f:
        image, mask = line.strip("\n").split(' ')
        images.append(os.path.join(image_dir,image))
        masks.append(os.path.join(label_dir,mask))
        pred_regname.append(mask)          
    return images, masks, pred_regname   


## read image and label
def read_image_from_disk(img_filename,label_filename,ii):
    img3 = np.zeros((321,321,3))
    str_convert = ''.join(img_filename[ii])
    img = misc.imread(str_convert)
    img = misc.imresize(img,[321,321])
    img = img.astype("float32")
    max_ = np.amax(img)
    min_ = np.amin(img)
    img = 255*(img - min_) / (max_ - min_)
    img3[:,:,0]=img
    img3[:,:,1]=img
    img3[:,:,2]=img
    str_convert = ''.join(label_filename[ii])
    mat = io.loadmat(str_convert)
    label_image0 = mat["Blabel"]
    label_image = misc.imresize(label_image0,[321,321],interp="nearest")
    label_image321 = label_image
    dislab1 = ndi.distance_transform_edt(label_image)
    dislab2 = ndi.distance_transform_edt(1-label_image)
    dislab = dislab1
    dislab[label_image==0] = dislab2[label_image==0]
    dislab = np.exp(-lamda*(dislab-1))
    dislab321=dislab
    return img3.astype("float32"),label_image321.astype("float32"),dislab321.astype("float32")





    
def main():
    "Create the model and start the evaluation process."
    #args = get_arguments()
    image_name, mask_name, pred_regnames  = read_pred_label_list(IMAGE_PATH,LABEL_PATH,TEST_LIST_PATH)
    image_batch = np.zeros((batch_size,321,321,3))
    trainpred_distanceB = np.zeros((batch_size,321,321))
    trainlabel_distance321B = np.zeros((batch_size,321,321))
    trainmask_batch = np.zeros((batch_size,321,321)) 
    for ii in range(batch_size):
         image_batch[ii,:,:,:],trainmask_batch[ii,:,:],trainlabel_distance321B[ii,:,:]= read_image_from_disk(image_name,mask_name,ii)
    image_batch = np.reshape(image_batch,(batch_size, 321, 321, 3)) 
    trainpred_distanceB = np.reshape(trainpred_distanceB,(batch_size, 321, 321, 1))
    trainlabel_distance321B = np.reshape(trainlabel_distance321B,(batch_size, 321, 321, 1))
    trainmask_batch = np.reshape(trainmask_batch,(batch_size, 321, 321, 1))
    ind = tf.placeholder(tf.int32, shape=(1, 1))
    img_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    print("img_batch"+repr(img_batch)) 
    trainpred_distanceB = tf.convert_to_tensor(trainpred_distanceB, dtype=tf.float32)
    trainlabel_distance321B = tf.convert_to_tensor(trainlabel_distance321B, dtype=tf.float32)
    trainmask_batch = tf.convert_to_tensor(trainmask_batch, dtype=tf.float32)
    img_slice = tf.py_func(image_slice, [img_batch,ind], tf.float32)
    print("img_slice"+repr(img_slice))  
    trainpred_distance = tf.py_func(image_slice, [trainpred_distanceB,ind], tf.float32)
    trainlabel_distance321 = tf.py_func(image_slice, [trainlabel_distance321B,ind], tf.float32)
    trainmask321 = tf.py_func(image_slice, [trainmask_batch,ind], tf.float32)
    #Create network.
    net_deeplab = DeepLabLFOVModel(WEIGHTS_PATH)  ##transfer learning for regression network   
    net_regression = RegressionNet()## boundary distance regression network
    net_seg = DeepLabSEGModel(WEIGHTS_PATH)## pixelwise classification network
    _,_,_,train_feature1024 = net_deeplab.preds(img_slice,1)
    trainpred_regression = net_regression.preds(train_feature1024)
    trainpred_input = tf.concat([trainpred_regression*255, trainpred_regression*255, trainpred_regression*255], 3)
    ## compute index
    dice_M = dice_mask(tf.cast(trainmask_segmentation,tf.float32), trainmask321)

    #Which variable to load
    trainable = tf.trainable_variables()
    #print('trainable'+repr(trainable)) 
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    # Load weights
    saver = tf.train.Saver(var_list=trainable)
    saver.restore(sess, model_weights)
        # Perform inference
    if not os.path.exists(save_pred_dir):
        os.makedirs(save_pred_dir)
    for ii in range(batch_size):
        preds,masks,regression,M_dice = sess.run([trainmask_segmentation,trainmask_regression,trainpred_input,dice_M],feed_dict={ind: np.reshape(ii,(1,1))})
        io.savemat(save_pred_dir+''.join(pred_regnames[ii]), {'predmask': preds})
        dice_M_matrix[ii,:]=M_dice
    dice_M_mean = np.mean(dice_M_matrix)
    print("dice_M_mean"+repr(dice_M_mean))
    dice_M_std = np.std(dice_M_matrix)
    print("dice_M_std"+repr(dice_M_std))


            
    
if __name__ == '__main__':
    main()