# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from skimage.measure import label, regionprops
from skimage.filters import sobel
import SimpleITK as sitk
from glob import glob
# import os
import re
import gc

import tensorflow as tf
import tensorlayer as tl
from keras.utils import to_categorical
import keras.backend.tensorflow_backend as K

# ----------------------------------------------------------------------------
# an argument to your Session's config arguments, it might help a little, 
# though it won't release memory, it just allows growth at the cost of some memory efficiency
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
# with tf.Session(config = config) as s:
sess = tf.Session(config = config)
K.set_session(sess)
# ----------------------------------------------------------------------------

nclasses = 5

def convert(str):
    return int("".join(re.findall("\d*", str)))

# copied from https://github.com/zsdonghao/u-net-brain-tumor
def distort_imgs(data):
    """ data augmentation """
    x1, x2, x3, x4,  y = data
    
    # previous without this, hard-dice=83.7
    x1, x2, x3, x4,  y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y], 
                                              axis=0, is_random=True) # up down
    x1, x2, x3, x4,  y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y], 
                                              axis=1, is_random=True) # left right
    x1, x2, x3, x4,  y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y], 
                                                      alpha=720, sigma=24, is_random=True)
    x1, x2, x3, x4,   y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20, 
                                             is_random=True, fill_mode='constant') # nearest, constant
    x1, x2, x3, x4,  y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10, 
                                          hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4,  y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05, 
                                          is_random=True, fill_mode='constant')
    x1, x2, x3, x4,  y = tl.prepro.zoom_multi([x1, x2, x3, x4, y], zoom_range=[0.9, 1.1], 
                                         is_random=True, fill_mode='constant')
    
    return x1, x2, x3, x4,  y

def read_scans(file_path1, data_tr_test=False):
    # nfiles = len(file_path1)
    scan_idx = 0
    nda_sum = 0
    for name in file_path1:
        # print ('\t', name)
        file_scan = sitk.ReadImage(name) # (240, 240, 155) = (rows, cols, slices)
        # if scan_idx == 99:
            # print ('\t', name)
        nda = sitk.GetArrayFromImage(file_scan) # convert to numpy array, (155, 240, 240)
        
        if scan_idx == 0:
            nda_sum = nda            
            # print(gt_sum.shape)
        else:
            # nda_sum = np.append(nda_sum, nda, axis=0)
            nda_sum = np.concatenate((nda_sum, nda), axis=0) # faster
            # print(nda_sum.shape)
        
        # scan_idx += 1
        if scan_idx < 99: # for BRATS 2015
            scan_idx += 1
        else:
            break
    
    #print(nda_sum.shape)
    return nda_sum


# def resize_data(imgs_train1, imgs_train2, imgs_train3, imgs_train4, imgs_label):
def resize_data(imgs_train1, imgs_train2, imgs_train3, imgs_label):
    # prepare data for CNNs with the softmax activation
    nslices = 0
    for n in range(imgs_train1.shape[0]):
        label_temp = imgs_label[n] # imgs_label[n][:][:]
        edges = sobel(label_temp)
        # print(label_temp.shape)
        c = np.count_nonzero(edges)
        # print(c)   
        if c > 149:
            train_resz1 = imgs_train1[n] # keep the original size of data
            train_resz2 = imgs_train2[n]
            train_resz3 = imgs_train3[n]
            # train_resz4 = imgs_train4[n]
            train_resz1 = train_resz1[..., np.newaxis]
            train_resz2 = train_resz2[..., np.newaxis]
            train_resz3 = train_resz3[..., np.newaxis]
            # train_resz4 = train_resz4[..., np.newaxis]
            
            train_sum = np.concatenate((train_resz1, train_resz2, train_resz3), axis=-1)
            # train_sum = np.concatenate((train_sum, train_resz3), axis=-1) # 240, 240, 3
            # train_sum = np.concatenate((train_sum, train_resz4), axis=-1) # 240, 240, 4
            
            train_sum = train_sum[np.newaxis, ...] # 1, 240, 240, 3
                        
            label_resz = label_temp
            label_resz2 = np.reshape(label_resz, 240*240).astype('int32')
            label_resz2 = to_categorical(label_resz2, nclasses)
                       
            label_resz2 = label_resz2[np.newaxis, ...] # 1, 240*240, nclasses
            if nslices == 0:
                # flair_sum = np.asarray([flair_resz]) same as np.reshape(label_resz, (1, 64, 64))
                # gt_sum = np.asarray([gt_resz])
                data_sum = train_sum
                label_sum = label_resz2
            else:                
                data_sum = np.concatenate((data_sum, train_sum), axis=0) # faster                
                label_sum = np.concatenate((label_sum, label_resz2), axis=0)
            
            nslices += 1
    # print(train_sum.shape)
    return data_sum, label_sum

# def resize_data(imgs_train1, imgs_train2, imgs_train3, imgs_train4, imgs_label):
def resize_data_aug(imgs_train1, imgs_train2, imgs_train3,imgs_train4, imgs_label):
    # prepare data for CNNs with the softmax activation
    # concat the original and augmented data
    nslices = 0
    for n in range(imgs_train1.shape[0]):
        label_temp = imgs_label[n] # imgs_label[n][:][:]; n, 240, 240
        edges = sobel(label_temp)
        print("slice :",n)
        c = np.count_nonzero(edges)
        print("****",c)   
        if c > 149:
            train_resz1 = imgs_train1[n] # keep the original size of data
            train_resz2 = imgs_train2[n]
            train_resz3 = imgs_train3[n]
            train_resz4 = imgs_train4[n]
            train_resz1 = train_resz1[..., np.newaxis]
            train_resz2 = train_resz2[..., np.newaxis]
            train_resz3 = train_resz3[..., np.newaxis]
            train_resz4 = train_resz4[..., np.newaxis]
            label_temp2 = label_temp.astype('int32')
            label_temp2 = label_temp2[..., np.newaxis]
            
            train_aug1, train_aug2, train_aug3, train_aug4, label_aug = distort_imgs([train_resz1, train_resz2, train_resz3,train_resz4, label_temp2])
            # print(train_aug1.shape, label_aug.shape) # 240, 240, 1
            
            train_sum = np.concatenate((train_resz1, train_resz2, train_resz3,train_resz4), axis=-1) # 240, 240, 3                      
            train_sum = train_sum[np.newaxis, ...] # 1, 240, 240, 3
            
            train_sum2 = np.concatenate((train_aug1, train_aug2, train_aug3,train_aug4), axis=-1) # 240, 240, 3
            train_sum2 = train_sum2[np.newaxis, ...] # 1, 240, 240, 3
                        
            # label_resz = label_temp
            label_resz = np.reshape(label_temp, 240*240).astype('int32')
            label_resz = to_categorical(label_resz, nclasses)
            label_resz = label_resz[np.newaxis, ...] # 1, 240*240, nclasses
            
            label_aug2 = label_aug[:, :, 0] # 240, 240, 1 to 240, 240
            label_resz2 = np.reshape(label_aug2, 240*240).astype('int32')
            label_resz2 = to_categorical(label_resz2, nclasses)                       
            label_resz2 = label_resz2[np.newaxis, ...] # 1, 240*240, nclasses
            
            if nslices == 0:                
                data_sum = np.concatenate((train_sum, train_sum2), axis=0) # faster                
                label_sum = np.concatenate((label_resz, label_resz2), axis=0)
            else:                
                data_sum = np.concatenate((data_sum, train_sum, train_sum2), axis=0) # faster                
                label_sum = np.concatenate((label_sum, label_resz, label_resz2), axis=0)
            
            nslices += 1
    
    return data_sum, label_sum

# for testing data
#def resize_data_full(imgs_train1, imgs_train2, imgs_train3, imgs_label):
def resize_data_full(imgs_train1, imgs_train2, imgs_train3, imgs_train4, imgs_label):
    
    # prepare data for CNNs with the softmax activation
    for n in range(imgs_label.shape[0]):        
        label_resz = imgs_label[n]
        label_resz2 = np.reshape(label_resz, 240*240).astype('int32')
        label_resz2 = to_categorical(label_resz2, nclasses)  
        label_resz2 = label_resz2[np.newaxis, ...] # 1, 240*240, nclasses                             
            
        if n == 0:            
            label_sum = label_resz2             
        else:
            label_sum = np.concatenate((label_sum, label_resz2), axis=0)                
    
    imgs_train1 = imgs_train1[..., np.newaxis] # 155, 240, 240, 1
    imgs_train2 = imgs_train2[..., np.newaxis]
    imgs_train3 = imgs_train3[..., np.newaxis]
    imgs_train4 = imgs_train4[..., np.newaxis]
    # concat data to 155, 240, 240, 3
    data_sum = np.concatenate((imgs_train1, imgs_train2, imgs_train3,imgs_train4), axis=-1)
        
    return data_sum, label_sum

def create_train_data(type_data='HGG'):
  
    
    if type_data == 'HGG':
        flairs = glob(r'mhafiles/BRATS2015_Training/HGG/*/*Flair*/*Flair*.mha')
        t1cs = glob(r'mhafiles/BRATS2015_Training/HGG/*/*T1c*/*T1c*.mha')
        t2s = glob(r'mhafiles/BRATS2015_Training/HGG/*/*T2*/*T2*.mha')
        t1s = glob('mhafiles/BRATS2015_Training/HGG/*/*T1*/*T1.*.mha')
        gts = glob('mhafiles/BRATS2015_Training/HGG/*/*OT*/*OT*.mha')

    
    flairs.sort(key=convert)
    t1cs.sort(key=convert)
    t2s.sort(key=convert)
    # t1s.sort(key=convert)
    gts.sort(key=convert)
    nfiles = len(flairs)
    
    flair_sum = read_scans(flairs, True)
    print(flair_sum.shape)
    t1c_sum = read_scans(t1cs, True)
    print(t1c_sum.shape)
    t2_sum = read_scans(t2s, True)
    print(t2_sum.shape)
    t1_sum = read_scans(t1s, True)
    print(t1_sum.shape)
    gt_sum = read_scans(gts)
    print(gt_sum.shape)
    
    print('Combining training data for the softmax activation...')
    total3_train, gt_train = resize_data_aug(flair_sum, t1c_sum, t2_sum,t1_sum, gt_sum)
    print(total3_train.shape)
    print(gt_train.shape)
    
    if type_data == 'HGG':      
        # full HGG data of BRATS 2013
        np.save('mhafiles/Data/imgs_train_unet_HG.npy', total3_train)
        np.save('mhafiles/Data/imgs_label_train_unet_HG.npy', gt_train)
        # np.save('D:\mhafiles\Data\imgs_train_unet_IN.npy', total3_train)
        # np.save('D:\mhafiles\Data\imgs_label_train_unet_IN.npy', gt_train)
        print('Saving all HGG training data to .npy files done.')          
    else:
        print('Cannot save type of data as you want')   
    
    for i in range(30):
        gc.collect()

def load_train_data(type_data='HGG'):
    imgs_label=0
    imgs_train=0
    if type_data == 'HGG':       
        imgs_train = np.load('mhafiles/Data/imgs_train_unet_HG.npy')
        imgs_label = np.load('mhafiles/Data/imgs_label_train_unet_HG.npy')
        print('Imgs train shape', imgs_train.shape)  
        print('Imgs label shape', imgs_label_train.shape)
        # imgs_train = np.load('D:\mhafiles\Data\imgs_train_unet_IN.npy')
        # imgs_label = np.load('D:\mhafiles\Data\imgs_label_train_unet_IN.npy')
    elif type_data == 'LGG':
        imgs_train = np.load('mhafiles/Data/imgs_train_unet_LG.npy')
        imgs_label = np.load('mhafiles/Data/imgs_label_train_unet_LG.npy')
    elif type_data == 'Full_HGG':
        imgs_train = np.load('mhafiles\Data/imgs_train_unet_FHG.npy')
        imgs_label = np.load('mhafiles/Data/imgs_label_train_unet_FHG.npy')
    else:
        print('No type of data as you want')
        
    return imgs_train, imgs_label
    
def create_test_data():    
    # flairs_test = glob('D:\mhafiles\HGG_Flair_6.mha')
    # t1cs_test = glob('D:\mhafiles\HGG_T1c_6.mha')
    # t2s_test = glob('D:\mhafiles\HGG_T2_6.mha')
    # gts = glob('D:\mhafiles\HGG_OT_6.mha')
    
    # BRATS 2015
    flairs_test = glob(r'mhafiles/BRATS2015_Training/HGG/*2013*/*Flair*/*Flair*.mha')
    t1cs_test = glob(r'mhafiles/BRATS2015_Training/HGG/*2013*/*T1c*/*T1c*.mha')
    t2s_test = glob(r'mhafiles/BRATS2015_Training/HGG/*2013*/*T2*/*T2*.mha')
    t1s_test = glob('mhafiles/BRATS2015_Training/HGG/*2013*/*T1.*/*T1.*.mha')
    gts = glob('mhafiles/BRATS2015_Training/HGG/*2013*/*OT*/*OT*.mha') 
                
    flair_sum = read_scans(flairs_test, True)
    # flair_sum = read_scans_IN(flairs_test, 0)
    # flair_sum = read_scans_Nyul(flairs_test, 0)
    # flair_sum = read_scans_Nyul_IN(flairs_test, 0)
    print(flair_sum.shape)
    t1c_sum = read_scans(t1cs_test, True)
    # t1c_sum = read_scans_IN(t1cs_test, 1)
    # t1c_sum = read_scans_Nyul(t1cs_test, 1)
    # t1c_sum = read_scans_Nyul_IN(t1cs_test, 1)
    print(t1c_sum.shape)
    t2_sum = read_scans(t2s_test, True)
    # t2_sum = read_scans_IN(t2s_test, 2)
    # t2_sum = read_scans_Nyul(t2s_test, 2)
    # t2_sum = read_scans_Nyul_IN(t2s_test, 2)
    print(t2_sum.shape)
    t1_sum = read_scans(t1s_test, True)
    # t1_sum = read_scans_IN(t1s_test, 2)
    print(t1_sum.shape)
    gt_sum = read_scans(gts)
    print(gt_sum.shape)
    
    print('Resizing testing data for the softmax activation...')
    total3_test, gt_test = resize_data_full(flair_sum, t1c_sum, t2_sum, t1_sum, gt_sum)
    print(total3_test.shape)
    print(gt_test.shape)
        
    np.save('mhafiles/Data/imgs_test_unet_HN.npy', total3_test)
    np.save('mhafiles/Data/imgs_label_test_unet_HN.npy', gt_test)   
    print('Saving testing data.')
    
    for i in range(30):
        gc.collect()     
   

def load_test_data():
    imgs_test = np.load('mhafiles/Data/imgs_test_unet_HN.npy')
    imgs_label_test = np.load('mhafiles/Data/imgs_label_test_unet_HN.npy')

    return imgs_test, imgs_label_test

def load_val_data():
    imgs_val = np.load('mhafiles/Data/imgs_val_unet.npy')
    imgs_label_val = np.load('mhafiles/Data/imgs_label_val_unet.npy')    
            
    return imgs_val, imgs_label_val

if __name__ == '__main__':
    create_train_data('HGG')
    #create_test_data() 
    #load_train_data('HGG')