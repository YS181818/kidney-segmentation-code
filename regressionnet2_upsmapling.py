## The boundary distance regression network
import os
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc as misc
import sys
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

    
class RegressionNet(object):
        
    def create_network(self, input_batch, batch_size0):
        inter_layer = self.create_conv(input_batch,1,64*3,5,1,batch_size0)
        inter_layer = self.create_conv(inter_layer,2,32*3,5,2,batch_size0)
        inter_layer = self.create_conv(inter_layer,3,16*3,5,2,batch_size0)
        inter_layer = self.create_conv(inter_layer,4,1,5,2,batch_size0)
         
        return inter_layer


    def create_conv(self,lastlayer_output,conv_id,filter_nums,kernel_size,stride,batch_size0):
        with tf.name_scope("conv%d" % conv_id ):
            layer = "conv%d" % conv_id
            if layer == "conv1":
                w,b = self.get_weights_and_biases(layer,filter_nums,kernel_size)
                layer_output = tf.nn.conv2d(lastlayer_output, w, strides=[1,stride,stride,1], padding="SAME", name=layer) + b
            else: 
                w,b,outshape = self.get_weights_and_biases_trans(layer,filter_nums,kernel_size,batch_size0)
                layer_output = tf.nn.conv2d_transpose(lastlayer_output, w, outshape, strides=[1,stride,stride,1], padding="SAME", name=layer) + b
            lastlayer_output = layer_output
            layer = "relu%d" % conv_id
            layer_output = tf.nn.relu(lastlayer_output,name=layer)
            lastlayer_output = layer_output
            print("lastlayer_output"+repr(lastlayer_output))
        return lastlayer_output
   
    def get_weights_and_biases(self,layer,filter_nums,kernel_size):
        w_shape = [kernel_size,kernel_size,1,filter_nums]
        b_shape = [filter_nums]    
        with tf.variable_scope("variable",reuse=tf.AUTO_REUSE):
            initializer = tf.random_normal_initializer(0,0.001)
            w = tf.get_variable(name="%s_w" % layer,shape=w_shape,initializer=initializer)
            initializer = tf.constant_initializer(0)
            b = tf.get_variable(name="%s_b" % layer,shape=b_shape,initializer=initializer)


        return w,b
   
   
   
    def get_weights_and_biases_trans(self,layer,filter_nums,kernel_size,batch_size0):
        w_shape = [kernel_size,kernel_size,filter_nums,1]
        b_shape = [filter_nums]
        if layer == "conv2":
            w_shape[3] = 64*3
            outshape = [batch_size0,81,81,filter_nums]
        if layer == "conv3":
            w_shape[3] = 32*3
            outshape = [batch_size0,161,161,filter_nums]
        if layer == "conv4":
            w_shape[3] = 16*3
            outshape = [batch_size0,321,321,filter_nums]
        

        with tf.variable_scope("variable",reuse=tf.AUTO_REUSE):
            initializer = tf.random_normal_initializer(0,0.001)
            w = tf.get_variable(name="%s_w" % layer,shape=w_shape,initializer=initializer)
            initializer = tf.constant_initializer(0)
            b = tf.get_variable(name="%s_b" % layer,shape=b_shape,initializer=initializer)


        return w,b,outshape
   
    def getloss(self, input_batch, label_batch, batch_size):
        output_batch = self.create_network(input_batch, batch_size)
        loss = tf.nn.l2_loss(output_batch-label_batch)
        return loss
        
    def preds(self, input_batch, batch_size):
        output_batch = self.create_network(input_batch, batch_size)
        return output_batch
