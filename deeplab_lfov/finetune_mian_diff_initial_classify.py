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


print("11")

#os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


# In[11]:


import tensorflow as tf
from six.moves import cPickle
# two path two conv multi-scale ship fuse

# Loading net skeleton with parameters name and shapes.
#with open("./util/net_skeleton.ckpt", "rb") as f:
#    net_skeleton = cPickle.load(f)

# The DeepLab-LargeFOV model can be represented as follows:
## input -> [conv-relu](dilation=1, channels=64) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=128) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=256) x 3 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=512) x 3 -> [max_pool](stride=1)
##       -> [conv-relu](dilation=2, channels=512) x 3 -> [max_pool](stride=1) -> [avg_pool](stride=1)
##       -> [conv-relu](dilation=12, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=21) -> [pixel-wise softmax loss].
n_classes = 2

# diffrent training stage
# 0 for init training, 1 for cascade training
training_stage_index = 0

#
training_stage_index_placeholder = tf.placeholder(tf.int16)

# all feed dict
all_feed_dict = {training_stage_index_placeholder:training_stage_index}

# all observed variables for detecting the training strategy
all_observed_variables = []

class DeepLabLFOVModel(object):
    def __init__(self, weights_path=None):
        self.l2_loss = 0
        if weights_path is not None:
            self.trained_weights = np.load(weights_path,encoding="latin1").item()
        else:
            self.trained_weights = None
        
    def _restore_weights_and_biases_for_pretrain(self,name):
        assert self.trained_weights is not None, "no trained weights available!"
        stack_index = int(name[5])
        layer_index = int(name[12])
        layer_name = "conv%d_%d" % (stack_index+1, layer_index)
        return self.trained_weights[layer_name]["weights"],self.trained_weights[layer_name]["biases"]
        
    def _get_weights_and_biases(self,name,shape=None,trainable=True):
        if name.startswith("stack"): # the weights for baseline network
            stack_index = int(name[5]) # note: starts with 0
            layer_index = int(name[12]) # note: starts with 1
            if stack_index < 5:
                with tf.variable_scope("variable",reuse=tf.AUTO_REUSE) as scope:
                    w,b = self._restore_weights_and_biases_for_pretrain(name)
                    init_w = tf.constant_initializer(w)
                    shape_w = w.shape
                    w = tf.get_variable(name="%s_w"%name,initializer=init_w,shape=shape_w)
                    init_b = tf.constant_initializer(b)
                    shape_b = b.shape
                    b = tf.get_variable(name="%s_b"%name,initializer=init_b,shape=shape_b)
                    self.weights[name] = [w,b]
                    self.l2_loss += tf.nn.l2_loss(w)
                    return w,b
            if stack_index < 6:
                shape = [3,3]
            else:
                shape = [1,1]
            channels = [64,128,256,512,512,1024,1024,n_classes]
            # get last layer's output channel
            if layer_index == 1: # the first layer of a stack
                if stack_index == 0: # the first stack
                    shape.append(3)
                else:
                    shape.append(channels[stack_index-1])
            else:
                shape.append(channels[stack_index])
            # get this layer's output channel
            shape.append(channels[stack_index])

        with tf.variable_scope("variable",reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable(name="%s_w" % name,shape=shape,initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32))
            b = tf.get_variable(name="%s_b" % name,shape=[shape[-1]],initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32))
        self.weights[name] = [w,b]
        self.l2_loss += tf.nn.l2_loss(w)
        return w,b
    
    def _create_network(self,input_batch,keep_prob):
        self.weights = {}

        # building network
        output, output512, output1024  = self._create_baseline(input_batch,keep_prob)

        return output, output512, output1024

    def _create_baseline(self,input_batch,keep_prob,trainable=True):
        with tf.name_scope("baseline") as scope:
            print("input_batch:%s" % repr(input_batch))
            current = self._create_stack(input_batch, stack_list=[1,1], stack_index=0,pool_size=3,pool_stride=2)
            current = self._create_stack(current, stack_list=[1,1], stack_index=1,pool_size=3,pool_stride=2)
            current = self._create_stack(current, stack_list=[1,1,1], stack_index=2,pool_size=3,pool_stride=2)
            current = self._create_stack(current, stack_list=[1,1,1], stack_index=3,pool_size=3,pool_stride=1)
            current = self._create_stack(current, stack_list=[2,2,2], stack_index=4,pool_size=3,pool_stride=1)
            current512 = tf.reduce_mean(current, axis=3)
            current512 = tf.expand_dims(current512, -1)
            current = self._create_stack(current, stack_list=[12], stack_index=5,keep_prob=keep_prob)
            current1024 = tf.reduce_mean(current, axis=3)
            current1024 = tf.expand_dims(current1024, -1)
            current = self._create_stack(current, stack_list=[1], stack_index=6,keep_prob=keep_prob)
            current = self._create_stack(current, stack_list=[1], stack_index=7)

        return current, current512, current1024
            
    def _create_stack(self,current,stack_list,stack_index,pool_size=None,pool_stride=None,keep_prob=None,trainable=True):
        with tf.name_scope("stack%d" % stack_index) as scope:
            layer_index = 1
            for layer_kind in stack_list:
                w,b = self._get_weights_and_biases("stack%d_layer%d" % (stack_index,layer_index))
                if layer_kind == 1:
                    current = tf.nn.conv2d(current,w,strides=[1,1,1,1],padding="SAME")
                else:
                    current = tf.nn.atrous_conv2d(current,w,2,padding="SAME")
                current = tf.layers.batch_normalization(current+b)
                current = tf.nn.relu(current)
                layer_index += 1
            if pool_size is not None:
                with tf.name_scope("pool") as scope:
                    current = tf.nn.max_pool(current,ksize=[1,pool_size,pool_size,1],strides=[1,pool_stride,pool_stride,1],padding="SAME")
                    if stack_index == 4:
                        current = tf.nn.avg_pool(current,ksize=[1,pool_size,pool_size,1],strides=[1,pool_stride,pool_stride,1],padding="SAME")
            with tf.name_scope("dropout") as scope:
                if keep_prob is not None:
                    current = tf.nn.dropout(current,keep_prob=keep_prob)
            return current

    def prepare_label(self, input_batch, new_size):
        with tf.name_scope('label_encode'):
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size[1:3]) # As labels are integer numbers, need to use NN interp.
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # Reducing the channel dimension.
            input_batch = tf.one_hot(input_batch, depth=n_classes)
        return input_batch

    def preds(self,input_batch):
        output,output512,output1024 = self._create_network(input_batch, keep_prob=tf.constant(1.0))
        input_batch_size = tf.shape(input_batch)
        pred,densitys = self.preds_single(output,input_batch_size)
        return  pred,densitys,output512,output1024
        
    def preds_single(self, raw_output, input_batch_size):
        raw_output = tf.image.resize_bilinear(raw_output, input_batch_size[1:3,])
        raw_output = tf.nn.softmax(raw_output)
        raw_output_max = tf.reduce_max(raw_output,axis=3)
        raw_output_max = tf.expand_dims(raw_output_max,dim=3)
        low_p = 0
        raw_output_max_p = []
        for high_p in [0.99911+i/10000 for i in range(10)]:
            raw_output_max_less_than_p = tf.less(raw_output_max,high_p)
            raw_output_max_big_than_p = tf.greater_equal(raw_output_max,low_p)
            raw_output_max_p_ = tf.logical_and(raw_output_max_big_than_p,raw_output_max_less_than_p)
            raw_output_max_p.append(tf.cast(raw_output_max_p_,tf.uint8))
            low_p = high_p

        raw_output = tf.argmax(raw_output, axis=3)
        raw_output = tf.expand_dims(raw_output, dim=3) # Create 4D-tensor.
        return tf.cast(raw_output, tf.uint8),raw_output_max_p

    def loss(self,img_batch,label_batch):
        output,_,_ = self._create_network(img_batch,keep_prob=tf.constant(1.0))
        output_ = tf.reshape(output,[-1,n_classes])
        
        label = self.prepare_label(label_batch, tf.stack(output.get_shape()))
        loss = self.loss_single(output_,label)
        
        return loss
    
    def loss_single(self, pred_batch, label_batch):
        gt = tf.reshape(label_batch, [-1, n_classes])
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=pred_batch)
        reduced_loss = tf.reduce_sum(loss)
            
        return reduced_loss
    
def dice_m(input_batch1, input_batch2, num):

    input_1=tf.equal(input_batch1, num, name=None)
    print ("input_1:"+repr(input_1))
    input_2=tf.equal(input_batch2, num, name= None)
    print ("input_2:"+repr(input_2))
    accuracy_n=tf.logical_and(input_1, input_2,name=None)
    print ("accuracy_n:"+repr(accuracy_n))
    input2_z=tf.cast(input_2, tf.int32)
    input1_z=tf.cast(input_1, tf.int32)
    accuracy_z=tf.cast(accuracy_n, tf.int32)
    accuracy_n= 2*tf.reduce_sum(accuracy_z)/(tf.reduce_sum(input1_z)+tf.reduce_sum(input2_z))
    print ("accuracy:"+repr(accuracy_n))
    return accuracy_n

def density_estimate(pred,density_mask,label):
    density_mask = tf.greater(density_mask,0)
    right_pred = tf.equal(pred,label)
    right_density = tf.logical_and(density_mask,right_pred)
    all_density_estimate = tf.reduce_sum(tf.cast(density_mask,tf.float32))
    right_density_estimate = tf.reduce_sum(tf.cast(right_density,tf.float32))
    error_density = tf.logical_and(density_mask,tf.logical_not(right_density))
    return right_density_estimate, all_density_estimate-right_density_estimate,tf.cast(right_density,tf.uint8),tf.cast(error_density,tf.uint8)

print("21")






