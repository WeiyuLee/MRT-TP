import scipy
import scipy.io as sio
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import os
import preprocess as preprc

import pandas as pd 
import random

def _load_image(path, is_grayscale=False):
    
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)    
    
def load_data(dataset_folder_path, label_type):
    """
    Load the image data
    """
    
    data = {}
    label = {}
    
    for curr_label_type in label_type:
        
        file_name_list = os.listdir(os.path.join(dataset_folder_path, curr_label_type))
        print(len(file_name_list))
        if curr_label_type == "0":
            data[curr_label_type] = np.empty((len(file_name_list)*2, 360, 201, 1), float)        
            label[curr_label_type] = np.empty((len(file_name_list)*2, 1), int)        
        elif curr_label_type == "1":
            data[curr_label_type] = np.empty((len(file_name_list)*6, 360, 201, 1), float)        
            label[curr_label_type] = np.empty((len(file_name_list)*6, 1), int)        

        idx = 0
        
        for curr_file_name in file_name_list:
                
            curr_image = _load_image(os.path.join(dataset_folder_path, curr_label_type, curr_file_name), is_grayscale=True)
            curr_image = np.expand_dims(curr_image, axis=0)
            curr_image = np.expand_dims(curr_image, axis=-1)

            #data[curr_label_type] = np.append(data[curr_label_type], curr_image, axis=0)
            data[curr_label_type][idx] = curr_image
            label[curr_label_type][idx] = int(curr_label_type)           
            idx = idx + 1
            
            if curr_label_type == "0": # Balance normal data
                curr_image_ud = preprc.vertical_flip(curr_image)
                
                #data[curr_label_type] = np.append(data[curr_label_type], curr_image_ud, axis=0)
                #label[curr_label_type] = np.append(label[curr_label_type], [[int(curr_label_type)]], axis=0)                

                data[curr_label_type][idx] = curr_image_ud
                label[curr_label_type][idx] = int(curr_label_type)
                idx = idx + 1            
                
            if curr_label_type == "1": # Balance abnormal data
                curr_image_ud = preprc.vertical_flip(curr_image)
                curr_image_ud_lr = preprc.horizontal_flip(curr_image_ud)
                curr_image_lr = preprc.horizontal_flip(curr_image)
                curr_image_lr_ud = preprc.vertical_flip(curr_image_lr)
                curr_image_rot = preprc.rotate_180(curr_image)
                
#                data[curr_label_type] = np.append(data[curr_label_type], curr_image_ud, axis=0)
#                label[curr_label_type] = np.append(label[curr_label_type], [[int(curr_label_type)]], axis=0)                
#
#                data[curr_label_type] = np.append(data[curr_label_type], curr_image_ud_lr, axis=0)
#                label[curr_label_type] = np.append(label[curr_label_type], [[int(curr_label_type)]], axis=0)                
#
#                data[curr_label_type] = np.append(data[curr_label_type], curr_image_lr, axis=0)
#                label[curr_label_type] = np.append(label[curr_label_type], [[int(curr_label_type)]], axis=0)        
#
#                data[curr_label_type] = np.append(data[curr_label_type], curr_image_lr_ud, axis=0)
#                label[curr_label_type] = np.append(label[curr_label_type], [[int(curr_label_type)]], axis=0)        
#
#                data[curr_label_type] = np.append(data[curr_label_type], curr_image_rot, axis=0)
#                label[curr_label_type] = np.append(label[curr_label_type], [[int(curr_label_type)]], axis=0)        
                
                data[curr_label_type][idx] = curr_image_ud
                label[curr_label_type][idx] = int(curr_label_type)
                idx = idx + 1

                data[curr_label_type][idx] = curr_image_ud_lr
                label[curr_label_type][idx] = int(curr_label_type)
                idx = idx + 1

                data[curr_label_type][idx] = curr_image_lr
                label[curr_label_type][idx] = int(curr_label_type)
                idx = idx + 1
                
                data[curr_label_type][idx] = curr_image_lr_ud
                label[curr_label_type][idx] = int(curr_label_type)
                idx = idx + 1
                
                data[curr_label_type][idx] = curr_image_rot
                label[curr_label_type][idx] = int(curr_label_type)
                idx = idx + 1
                
            print(curr_file_name)
            
        print("[{}] {}: {}".format(curr_label_type, data[curr_label_type].shape, label[curr_label_type].shape))

    return data, label

def preprocess_and_save_data(dataset_folder_path, output_path, label_type):
    """
    Preprocess Training and Validation Data
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load training data ======================================================
    data, label = load_data(dataset_folder_path, label_type)
    pickle.dump((data, label), open(os.path.join(output_path, 'raw_data_label.p'), 'wb'), protocol=4)
        
    # Preprocess training & validation data
    train_data = np.empty((0, 360, 201, 1), float)       
    train_label = np.empty((0, 1), int)       
    test_data = np.empty((0, 360, 201, 1), float)       
    test_label = np.empty((0, 1), int)          
    for curr_label_type in label_type:
        print("[Before] {} data: ".format(curr_label_type), np.shape(data[curr_label_type]))
        
        idx = list(range(data[curr_label_type].shape[0]))
        random.shuffle(idx)
        train_sample_num = int(0.7*len(idx))
           
        train_data = np.append(train_data, data[curr_label_type][idx[1:train_sample_num]], axis=0)
        train_label = np.append(train_label, label[curr_label_type][idx[1:train_sample_num]], axis=0)

        test_data = np.append(test_data, data[curr_label_type][idx[train_sample_num:]], axis=0)
        test_label = np.append(test_label, label[curr_label_type][idx[train_sample_num:]], axis=0)

    train_data, _, _ = preprc.normalize(train_data, mean=[], std=[])
    test_data, _, _ = preprc.normalize(test_data, mean=[], std=[])

    # One-hot encode       
    train_label = preprc.one_hot_encode(train_label, dim=2)
    test_label = preprc.one_hot_encode(test_label, dim=2)

    print("[After] train_data shape: ", np.shape(train_data))
    print("[After] train_label shape: ", np.shape(train_label))
    print("[After] test_data shape: ", np.shape(test_data))
    print("[After] test_label shape: ", np.shape(test_label))
    
    # Save training data
    pickle.dump((train_data, train_label), open(os.path.join(output_path, 'preprocess_train.p'), 'wb'), protocol=4)
    pickle.dump((test_data, test_label), open(os.path.join(output_path, 'preprocess_test.p'), 'wb'), protocol=4)

# -----------------------------------------------------------------------------

dataset_folder_path = "/data/wei/dataset/MRT/"
output_path = "/data/wei/dataset/MRT/preprocessed/"

label = ["0", "1"]

preprocess_and_save_data(dataset_folder_path, output_path, label)
