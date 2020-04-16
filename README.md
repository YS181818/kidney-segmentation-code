Automatic kidney segmentation in ultrasound images using subsequent boundary distance regression and pixelwise classification networks-TensorFlow
This is an implementation of the proposed model in TensorFlow for kidney ultrasound image segmentation.

Model Description：
We first use deep neural networks pre-trained for classification of natural images to extract high-level image features from US images. 
These features are used as input to learn kidney boundary distance maps using a boundary distance regression network and the predicted boundary distance maps are classified as kidney pixels or non-kidney  pixels using a pixelwise classification network in an end-to-end learning fashion. 
We also adopted a data augmentation method based on kidney shape registration to generate enriched training data from a small number of US images with manually segmented kidney labels. 

For more details on the underlying model please refer to the following paper:
@article{yin2020automatic,
title={Automatic kidney segmentation in ultrasound images using subsequent boundary distance regression and pixelwise classification networks},
author={Yin, Shi and Peng, Qinmu and Li, Hongming and Zhang, Zhengqiang and You, Xinge and Fischer, Katherine and Furth, Susan L and Tasian, Gregory E and Fan, Yong},
journal={Medical Image Analysis},
volume={60},
pages={101602},
year={2020}}

We refer to the boundary distance regression network followed by post-processing for segmenting kidneys as a boundary detection network (Bnet)


Requirements：
The proposed networks were implemented based on Python 3.7.0 and TensorFlow r1.11


Training：
We initialized the network from the VGG_16.npy
Training the end-to-end learning of subsequent segmentation networks: train_end_to_end.py
Training the Bnet: train_Bnet.py


Evaluation:
Evaluating the end-to-end learning of subsequent segmentation networks: evaluate_end_to_end.py
Evaluating the Bnet: evaluate_Bnet.py

Data augmentation:
data augmentaion/main_preprocessing.m
The dataaugmentation code was implemented based on Matlab R2015b