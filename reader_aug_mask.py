## read image and label 
import os
import numpy as np
import scipy.misc as misc
import scipy.io as io
import tensorflow as tf
import scipy.ndimage as ndi
import math
ceil = math.ceil

class Image_Reader(object):
    def __init__(self, image_dir, label_dir, data_list, input_size):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.data_list = data_list
        
        self.image_list, self.label_list = self.read_pred_label_list(self.image_dir, self.label_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=input_size is not None) # Not shuffling if it is val.
        self.image_img, self.mask_img, self.label_img321 = tf.py_func(self.read_image_from_disk,self.queue,[tf.float32,tf.float32,tf.float32])
        self.image_img = tf.reshape(self.image_img,[input_size[0],input_size[1],3])
        self.label_img321 = tf.reshape(self.label_img321,[input_size[0],input_size[1],1])
        self.mask_img = tf.reshape(self.mask_img,[input_size[0],input_size[1],1])


    def dequeue(self, num_elements):
        image_img_batch, mask_img_batch, label_img321_batch= tf.train.batch([self.image_img,self.mask_img,self.label_img321],
                                                  num_elements)## train batch
        return image_img_batch, mask_img_batch, label_img321_batch

## read image and label list function
    def read_pred_label_list(self,image_dir,label_dir,data_list):
        f = open(data_list, 'r')
        images = []
        masks = []
        for line in f:
            image, mask = line.strip("\n").split(' ')
            images.append(os.path.join(image_dir,image))
            masks.append(os.path.join(label_dir,mask))           
        return images, masks   

## read image and label function
    def read_image_from_disk(self,img_filename,label_filename):
        img3 = np.zeros((321,321,3))
        img_filename = img_filename.decode()
        label_filename = label_filename.decode()
        img = misc.imread(img_filename)
        img = misc.imresize(img,[321,321])
        img = img.astype("float32")
        max_ = np.amax(img)
        min_ = np.amin(img)
        img = 255*(img - min_) / (max_ - min_)
        img3[:,:,0]=img
        img3[:,:,1]=img
        img3[:,:,2]=img
        str_convert = ''.join(label_filename)
        mat = io.loadmat(str_convert)
        label_image0 = mat["Blabel"]
        label_image = misc.imresize(label_image0,[321,321],interp="nearest")
        label_image321 = label_image
        dislab1 = ndi.distance_transform_edt(label_image)
        dislab2 = ndi.distance_transform_edt(1-label_image)
        dislab = dislab1
        dislab[label_image==0] = dislab2[label_image==0]
        dislab = np.exp(-1.0*(dislab-1))
        dislab321=dislab
        return img3.astype("float32"),label_image321.astype("float32"),dislab321.astype("float32")
    
