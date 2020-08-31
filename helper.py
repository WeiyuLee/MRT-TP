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
import imageio

class util(object):

    def __init__(self):
        return

    def _load_image(self, path, is_grayscale=False):
        
        if is_grayscale:
            return scipy.misc.imread(path, flatten=True).astype(np.float)
        else:
            return scipy.misc.imread(path, mode='RGB').astype(np.float)    

    def _augmentation(self, data, label, level=3):

        aug_func = [preprc.vertical_flip, preprc.horizontal_flip, preprc.rotate_180]

        file_num = data.shape[0]
        aug_data = np.empty((file_num*(level+1), 360, 201, 1), float)        
        aug_label = np.empty((file_num*(level+1), 1), int)      
        
        idx = 0
        for f in range(file_num):

            curr_image = data[[f], :, :, :]
            curr_label = label[[f], :]

            aug_data[idx] = curr_image
            aug_label[idx] = curr_label
            idx = idx + 1

            for lv in range(level):
                
                if level <= 1:
                    rdm = random.randint(0, len(aug_func)-1)
                    aug_image = aug_func[rdm](curr_image)
                
                else:                   
                    if lv <= 2:
                        aug_image = aug_func[lv](curr_image)
                    else: # gaussian_noise
                        rdm = random.randint(0, len(aug_func)-1)
                        aug_image = aug_func[rdm](curr_image)
                        gaussian_noise = np.random.normal(0, 0.05*255, curr_image.shape) 
                        aug_image = aug_image + gaussian_noise

                        #if f == 0:
                        #    imageio.imwrite("./output/origin.png", np.squeeze(curr_image))
                        #    imageio.imwrite("./output/noise.png", np.squeeze(aug_image))

                aug_data[idx] = aug_image
                aug_label[idx] = curr_label
                idx = idx + 1            

        return aug_data, aug_label

    def load_1st_data(self, dataset_folder_path, label_type):
        """
        Load the image data
        """

        if label_type == ["0", "1", "2"]:
            output_label = 0
        elif label_type == ["3"]:
            output_label = 1

        # Initial array
        file_num = 0
        for curr_label_type in label_type:
            file_name_list = os.listdir(os.path.join(dataset_folder_path, curr_label_type))
            file_num += len(file_name_list)

        data = np.empty((file_num, 360, 201, 1), float)        
        label = np.empty((file_num, 1), int)        

        idx = 0
            
        for curr_label_type in label_type:
            
            file_name_list = os.listdir(os.path.join(dataset_folder_path, curr_label_type))
            print(len(file_name_list))
            
            for curr_file_name in file_name_list:
                    
                curr_image = self._load_image(os.path.join(dataset_folder_path, curr_label_type, curr_file_name), is_grayscale=True)
                curr_image = np.expand_dims(curr_image, axis=0)
                curr_image = np.expand_dims(curr_image, axis=-1)

                data[idx] = curr_image
                label[idx] = output_label
                idx = idx + 1

                print(curr_file_name)
                
            print("[{}] {}: {}".format(curr_label_type, data.shape, label.shape))

        return data, label

    def load_data(self, dataset_folder_path, label_type):
        """
        Load the image data
        """
        
        data = {}
        label = {}
        
        for curr_label_type in label_type:
            
            file_name_list = os.listdir(os.path.join(dataset_folder_path, curr_label_type))
            print(len(file_name_list))
            if curr_label_type == "0":
                data[curr_label_type] = np.empty((len(file_name_list), 360, 201, 1), float)        
                label[curr_label_type] = np.empty((len(file_name_list), 1), int)        
            elif curr_label_type == "1":
                data[curr_label_type] = np.empty((len(file_name_list), 360, 201, 1), float)        
                label[curr_label_type] = np.empty((len(file_name_list), 1), int)        

            idx = 0
            
            for curr_file_name in file_name_list:
                    
                curr_image = self._load_image(os.path.join(dataset_folder_path, curr_label_type, curr_file_name), is_grayscale=True)
                curr_image = np.expand_dims(curr_image, axis=0)
                curr_image = np.expand_dims(curr_image, axis=-1)

                data[curr_label_type][idx] = curr_image
                label[curr_label_type][idx] = int(curr_label_type)           
                idx = idx + 1
                '''
                if curr_label_type == "0": # Balance normal data
                    curr_image_ud = preprc.vertical_flip(curr_image)
          
                    data[curr_label_type][idx] = curr_image_ud
                    label[curr_label_type][idx] = int(curr_label_type)
                    idx = idx + 1            
                    
                if curr_label_type == "1": # Balance abnormal data
                    curr_image_ud = preprc.vertical_flip(curr_image)
                    curr_image_ud_lr = preprc.horizontal_flip(curr_image_ud)
                    curr_image_lr = preprc.horizontal_flip(curr_image)
                    curr_image_lr_ud = preprc.vertical_flip(curr_image_lr)
                    curr_image_rot = preprc.rotate_180(curr_image)
                          
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
                '''    
                print(curr_file_name)
                
            print("[{}] {}: {}".format(curr_label_type, data[curr_label_type].shape, label[curr_label_type].shape))

        return data, label

def preprocess_and_save_1st_data(dataset_folder_path, output_path, aug=True):
    """
    Preprocess Training and Validation Data
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)   

    u = util()

    # Load training data ======================================================
    true_data, true_label = u.load_1st_data(os.path.join(dataset_folder_path, "crop_true"), ["0", "1", "2"])
    false_data, false_label = u.load_1st_data(os.path.join(dataset_folder_path, "crop_false"), ["3"]) 
    data = np.append(true_data, false_data, axis=0)
    label = np.append(true_label, false_label, axis=0)
    pickle.dump((data, label), open(os.path.join(output_path, 'raw_data_label.p'), 'wb'), protocol=4)
        
    print("[Before] data: ", np.shape(data))

    # Preprocess training & validation data
    train_data = np.empty((0, 360, 201, 1), float)       
    train_label = np.empty((0, 1), int)       
    test_data = np.empty((0, 360, 201, 1), float)       
    test_label = np.empty((0, 1), int)          
    
    idx = list(range(data.shape[0]))
    random.shuffle(idx)
    train_sample_num = int(0.7*len(idx))
       
    train_data = np.append(train_data, data[idx[1:train_sample_num]], axis=0)
    train_label = np.append(train_label, label[idx[1:train_sample_num]], axis=0)

    test_data = np.append(test_data, data[idx[train_sample_num:]], axis=0)
    test_label = np.append(test_label, label[idx[train_sample_num:]], axis=0)

    if aug == True:
        train_data, train_label = u._augmentation(train_data, train_label)

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
    
def preprocess_and_save_data(dataset_folder_path, output_path, label_type, aug=True):
    """
    Preprocess Training and Validation Data
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    u = util()

    # Load training data ======================================================
    data, label = u.load_data(os.path.join(dataset_folder_path, "crop_true"), label_type)
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

        if aug == True:
            if curr_label_type == "0":
                aug_train_data, aug_train_label = u._augmentation(data[curr_label_type][idx[1:train_sample_num]], label[curr_label_type][idx[1:train_sample_num]], level=1)
            if curr_label_type == "1":                
                aug_train_data, aug_train_label = u._augmentation(data[curr_label_type][idx[1:train_sample_num]], label[curr_label_type][idx[1:train_sample_num]], level=5)
            
            print("[After] {} data: ".format(curr_label_type), np.shape(aug_train_data))

        train_data = np.append(train_data, aug_train_data, axis=0)
        train_label = np.append(train_label, aug_train_label, axis=0)
        
        if curr_label_type == "0":
            test_data = np.append(test_data, data[curr_label_type][idx[-216:]], axis=0)
            test_label = np.append(test_label, label[curr_label_type][idx[-216:]], axis=0)

        if curr_label_type == "1":
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

# For 1st stage data
#dataset_folder_path = "/data/wei/dataset/MRT/"
#output_path = "/data/wei/dataset/MRT/preprocessed_1st/"
#preprocess_and_save_1st_data(dataset_folder_path, output_path)

## For 2nd stage data
dataset_folder_path = "/data/wei/dataset/MRT/"
output_path = "/data/wei/dataset/MRT/preprocessed_2nd/"
label = ["0", "1"]
preprocess_and_save_data(dataset_folder_path, output_path, label)
