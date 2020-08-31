# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('./utility')

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from utils import get_batch, mkdir_p
import model_zoo
import timeit

import imageio

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim

class model(object):

    #build model
    def __init__(self, batch_size, max_iters, dropout, model_path, data_ob, log_dir, learnrate_init,
                 ckpt_name, test_ckpt, train_ckpt=[], restore_model=False, restore_step=0, class_num=17, model_ticket="none", output_inf_model=False):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.dropout = dropout
        self.saved_model_path = model_path
        self.data_ob = data_ob
        self.log_dir = log_dir
        self.learn_rate_init = learnrate_init
        self.ckpt_name = ckpt_name
        self.test_ckpt = test_ckpt
        self.train_ckpt = train_ckpt
        self.restore_model = restore_model
        self.restore_step = restore_step
        self.class_num = class_num
        self.model_ticket = model_ticket
        self.output_inf_model = output_inf_model
        
        self.output_dir = os.path.join("./output/", self.ckpt_name)
        mkdir_p(os.path.join(self.output_dir, "normal"))
        mkdir_p(os.path.join(self.output_dir, "abnormal"))
        mkdir_p(os.path.join(self.output_dir, "feature"))
                
        self.input = tf.placeholder(tf.float32, [self.batch_size, 360, 201, 1], name='input')
        self.label = tf.placeholder(tf.float32, [self.batch_size, 2], name='label')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout')
            
        ## Training set
        self.dataset = self.data_ob.train_data_list          

        ## Validation set
        self.valid_dataset = self.data_ob.valid_data_list

        ## Anomaly set
        self.anomaly_dataset = self.data_ob.anomaly_data_list
        
        ## Testing set
        self.test_dataset = self.data_ob.test_data_list
            
        self.model_list = ["CNN_v1", "CNN_1st_v1", "ResNet10_half", "ResNet10", "EXAMPLE_CNN"]

    def build_model(self):###              
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            print("Model name: {}".format(self.model_ticket))
            fn = getattr(self, "build_" + self.model_ticket)
            model = fn()
            return model    
        
    def build_eval_model(self):###              
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            print("Model name: {}".format(self.model_ticket))
            fn = getattr(self, "build_eval_" + self.model_ticket)
            model = fn()
            return model 
        
    def train(self):
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            fn = getattr(self, "train_" + self.model_ticket)
            function = fn()
            return function 

    def test(self):
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            print("Model name: {}".format(self.model_ticket))
            fn = getattr(self, "test_" + self.model_ticket)
            function = fn()
            return function 

    def build_CNN_1st_v1(self):
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Classifier =============================================================================================================
        logits, _ = mz.build_model({"input":self.input, "reuse":False})  

        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)
                                   
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label)) + self.reg_set_l2_loss
        self.pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        correct_pred = tf.equal(self.pred, tf.argmax(self.label, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))             

        train_variables = tf.trainable_variables()       
        slim.model_analyzer.analyze_vars(train_variables, print_info=True)
        var = [v for v in train_variables if v.name.startswith(("CNN"))]

        self.train_var = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss, var_list=var)
        
        with tf.name_scope('train_summary'):

            tf.summary.scalar("loss", self.loss, collections=['train'])           
            tf.summary.scalar("acc", self.acc, collections=['train'])           
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("loss", self.loss, collections=['test'])           
            tf.summary.scalar("acc", self.acc, collections=['test'])           
            
            self.merged_summary_test = tf.summary.merge_all('test')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_CNN_1st_v1(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        logits, conv_1 = mz.build_model({"input":self.input, "reuse":False})  

        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label)) + self.reg_set_l2_loss
        self.pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        correct_pred = tf.equal(self.pred, tf.argmax(self.label, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 

        self.normal_sample = tf.gather(self.input, tf.where(tf.equal(self.pred, 0)))
        self.ano_sample = tf.gather(self.input, tf.where(tf.equal(self.pred, 1)))
        
        self.feature_map = tf.reduce_mean(conv_1, axis=-1, keep_dims=True)

        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        
        self.saver = tf.train.Saver()

    def train_CNN_1st_v1(self):
        
        new_learning_rate = self.learn_rate_init
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            while step <= self.max_iters:
                
                # Get the training batch
                next_x, next_y = get_batch(self.dataset, self.batch_size)
                   
                fd = {
                        self.input: next_x, 
                        self.label: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                        self.is_training: True
                     }
                
                sess.run(self.train_var, feed_dict=fd) 

                # Update Learning rate                
                if step == 2000 or step == 4000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%100 == 0:

                    # Training set
                    train_sum, train_loss, train_acc = sess.run([self.merged_summary_train, self.loss, self.acc], feed_dict=fd)
                    
                    next_valid_x, next_valid_y = get_batch(self.valid_dataset, self.batch_size)

                    fd_test = {
                                self.input: next_valid_x, 
                                self.label: next_valid_y, 
                                self.dropout_rate: 0,
                                self.is_training: False
                              }
                                           
                    test_sum, test_loss, test_acc = sess.run([self.merged_summary_test, self.loss, self.acc], feed_dict=fd_test)  
                                          
                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Train acc = [%.7f], Test loss = [%.7f], Test acc = [%.7f]" % (step, new_learning_rate, train_loss, train_acc, test_loss, test_acc))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   

                    if abs(best_loss) > abs(test_loss):
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("\n* Save ckpt: {}, Test loss: {}\n".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step, 100) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_CNN_1st_v1(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
 
            if self.output_inf_model == True:
            
                # Save inference model (batch_size=1)===============================================================
                fd_test = {
                            self.input: self.test_dataset[0][0:1], 
                            self.label: self.test_dataset[1][0:1], 
                            self.dropout_rate: 0,
                          }
    
                curr_loss, curr_acc = sess.run([self.loss, self.acc], feed_dict=fd_test)               
    
                mkdir_p(os.path.join(self.saved_model_path, "inference"))
                save_path = self.saver.save(sess , os.path.join(self.saved_model_path, "inference", self.ckpt_name))
                print("Model saved in file: %s" % save_path)
                # ==================================================================================================

            else:
    
                start = timeit.default_timer()
        
                loss = 0
                acc = 0
        
                curr_idx = 0
                iteration = 0
                while True:
                    
                    try:
                        
                        if curr_idx >= len(self.test_dataset[0]):
                            curr_idx = curr_idx - self.batch_size
                            break
                        
                        next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                        next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                        
                        if len(next_x_images) < self.batch_size:
                            break
                        
                        curr_idx = curr_idx + self.batch_size
                        iteration = iteration + 1
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                    fd_test = {
                                self.input: next_x_images, 
                                self.label: next_test_y, 
                                self.dropout_rate: 0,
                                self.is_training: False
                              }
    
                    curr_loss, curr_acc = sess.run([self.loss, self.acc], feed_dict=fd_test)                   
                        
                    loss = loss + curr_loss
                    acc = acc + curr_acc
                
                stop = timeit.default_timer()
                
                loss = loss / iteration
                acc = acc / iteration
                
                print("Test loss: {}".format(loss))
                print("Test acc.: {}".format(acc))
                print("Time: {}".format(stop-start))
                print("Avg. Time: {}s per frame".format((stop-start)/curr_idx))

    def build_CNN_v1(self):
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Classifier =============================================================================================================
        logits, _ = mz.build_model({"input":self.input, "reuse":False})  

        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)
                                   
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label)) + self.reg_set_l2_loss
        self.pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        correct_pred = tf.equal(self.pred, tf.argmax(self.label, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))             

        train_variables = tf.trainable_variables()       
        var = [v for v in train_variables if v.name.startswith(("CNN"))]

        self.train_var = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss, var_list=var)
        
        slim.model_analyzer.analyze_vars(train_variables, print_info=True)

        with tf.name_scope('train_summary'):

            tf.summary.scalar("loss", self.loss, collections=['train'])           
            tf.summary.scalar("acc", self.acc, collections=['train'])           
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("loss", self.loss, collections=['test'])           
            tf.summary.scalar("acc", self.acc, collections=['test'])           
            
            self.merged_summary_test = tf.summary.merge_all('test')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_CNN_v1(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        logits, conv_1 = mz.build_model({"input":self.input, "reuse":False})  

        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label)) + self.reg_set_l2_loss
        self.pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        correct_pred = tf.equal(self.pred, tf.argmax(self.label, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 

        self.normal_sample = tf.gather(self.input, tf.where(tf.equal(self.pred, 0)))
        self.ano_sample = tf.gather(self.input, tf.where(tf.equal(self.pred, 1)))
        
        self.feature_map = tf.reduce_mean(conv_1, axis=-1, keep_dims=True)

        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        
        self.saver = tf.train.Saver()

    def train_CNN_v1(self):
        
        new_learning_rate = self.learn_rate_init
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            while step <= self.max_iters:
                
                # Get the training batch
                next_x, next_y = get_batch(self.dataset, self.batch_size)
                   
                fd = {
                        self.input: next_x, 
                        self.label: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                        self.is_training: True
                     }
                
                sess.run(self.train_var, feed_dict=fd) 

                # Update Learning rate                
                if step == 2000 or step == 4000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%100 == 0:

                    # Training set
                    train_sum, train_loss, train_acc = sess.run([self.merged_summary_train, self.loss, self.acc], feed_dict=fd)
                    
                    next_valid_x, next_valid_y = get_batch(self.valid_dataset, self.batch_size)

                    fd_test = {
                                self.input: next_valid_x, 
                                self.label: next_valid_y, 
                                self.dropout_rate: 0,
                                self.is_training: False
                              }
                                           
                    test_sum, test_loss, test_acc = sess.run([self.merged_summary_test, self.loss, self.acc], feed_dict=fd_test)  
                                          
                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Train acc = [%.7f], Test loss = [%.7f], Test acc = [%.7f]" % (step, new_learning_rate, train_loss, train_acc, test_loss, test_acc))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   

                    if abs(best_loss) > abs(test_loss):
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("\n* Save ckpt: {}, Test loss: {}\n".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step, 100) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_CNN_v1(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
 
            if self.output_inf_model == True:
            
                # Save inference model (batch_size=1)===============================================================
                fd_test = {
                            self.input: self.test_dataset[0][0:1], 
                            self.label: self.test_dataset[1][0:1], 
                            self.dropout_rate: 0,
                            self.is_training: False
                          }
    
                curr_loss, curr_acc = sess.run([self.loss, self.acc], feed_dict=fd_test)               
    
                mkdir_p(os.path.join(self.saved_model_path, "inference"))
                save_path = self.saver.save(sess , os.path.join(self.saved_model_path, "inference", self.ckpt_name))
                print("Model saved in file: %s" % save_path)
                # ==================================================================================================

            else:
    
                start = timeit.default_timer()
        
                loss = 0
                acc = 0
        
                curr_idx = 0
                iteration = 0
                while True:
                    
                    try:
                        
                        if curr_idx >= len(self.test_dataset[0]):
                            break
                        
                        next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                        next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                        
                        if len(next_x_images) < self.batch_size:
                            break
                        
                        curr_idx = curr_idx + self.batch_size
                        iteration = iteration + 1
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                    fd_test = {
                                self.input: next_x_images, 
                                self.label: next_test_y, 
                                self.dropout_rate: 0,
                              }
    
                    curr_loss, curr_acc, normal_img, ano_img, feature_img = sess.run([self.loss, self.acc, self.normal_sample, self.ano_sample, self.feature_map], feed_dict=fd_test)               
    
                    print(normal_img.shape)
                    print(ano_img.shape)
                    
                    for i in range(len(normal_img)):
                        imageio.imwrite("./output/normal/{}_{}.png".format(curr_idx, i), np.squeeze((normal_img[i]*255).astype(np.uint8)))
    
                    for i in range(len(ano_img)):
                        imageio.imwrite("./output/abnormal/{}_{}.png".format(curr_idx, i), np.squeeze((ano_img[i]*255).astype(np.uint8)))
    
                    for i in range(len(feature_img)):
                        imageio.imwrite("./output/feature/{}_{}.png".format(curr_idx, i), np.squeeze((feature_img[i])))
                    
                    loss = loss + curr_loss
                    acc = acc + curr_acc
                
                stop = timeit.default_timer()
                
                loss = loss / iteration
                acc = acc / iteration
                
                print("Test loss: {}".format(loss))
                print("Test acc.: {}".format(acc))
                print("Time: {}".format(stop-start))

    def build_ResNet10_half(self):
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Classifier =============================================================================================================
        logits, _ = mz.build_model({"input":self.input, "reuse":False})  

        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)
                                   
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label)) + self.reg_set_l2_loss
        self.pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        correct_pred = tf.equal(self.pred, tf.argmax(self.label, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))             

        train_variables = tf.trainable_variables()       
        var = [v for v in train_variables if v.name.startswith(("ResNet10"))]

        self.train_var = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss, var_list=var)

        slim.model_analyzer.analyze_vars(train_variables, print_info=True)

        with tf.name_scope('train_summary'):

            tf.summary.scalar("loss", self.loss, collections=['train'])           
            tf.summary.scalar("acc", self.acc, collections=['train'])           
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("loss", self.loss, collections=['test'])           
            tf.summary.scalar("acc", self.acc, collections=['test'])           
            
            self.merged_summary_test = tf.summary.merge_all('test')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_ResNet10_half(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        logits, conv_1 = mz.build_model({"input":self.input, "reuse":False})  

        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label)) + self.reg_set_l2_loss
        self.pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        correct_pred = tf.equal(self.pred, tf.argmax(self.label, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 

        self.normal_sample = tf.gather(self.input, tf.where(tf.equal(self.pred, 0)))
        self.ano_sample = tf.gather(self.input, tf.where(tf.equal(self.pred, 1)))
        
        self.feature_map = tf.reduce_mean(conv_1, axis=-1, keep_dims=True)

        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        
        self.saver = tf.train.Saver()

    def train_ResNet10_half(self):
        
        new_learning_rate = self.learn_rate_init
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            while step <= self.max_iters:
                
                # Get the training batch
                next_x, next_y = get_batch(self.dataset, self.batch_size)
                   
                fd = {
                        self.input: next_x, 
                        self.label: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                        self.is_training: True
                     }
                
                sess.run(self.train_var, feed_dict=fd) 

                # Update Learning rate                
                if step == 2000 or step == 4000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%100 == 0:

                    # Training set
                    train_sum, train_loss, train_acc = sess.run([self.merged_summary_train, self.loss, self.acc], feed_dict=fd)
                    
                    next_valid_x, next_valid_y = get_batch(self.valid_dataset, self.batch_size)

                    fd_test = {
                                self.input: next_valid_x, 
                                self.label: next_valid_y, 
                                self.dropout_rate: 0,
                                self.is_training: False
                              }
                                           
                    test_sum, test_loss, test_acc = sess.run([self.merged_summary_test, self.loss, self.acc], feed_dict=fd_test)  
                                          
                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Train acc = [%.7f], Test loss = [%.7f], Test acc = [%.7f]" % (step, new_learning_rate, train_loss, train_acc, test_loss, test_acc))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   

                    if abs(best_loss) > abs(test_loss):
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("\n* Save ckpt: {}, Test loss: {}\n".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step, 100) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_ResNet10_half(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
 
            if self.output_inf_model == True:
            
                # Save inference model (batch_size=1)===============================================================
                fd_test = {
                            self.input: self.test_dataset[0][0:1], 
                            self.label: self.test_dataset[1][0:1], 
                            self.dropout_rate: 0,
                            self.is_training: False
                          }
    
                curr_loss, curr_acc = sess.run([self.loss, self.acc], feed_dict=fd_test)               
    
                mkdir_p(os.path.join(self.saved_model_path, "inference"))
                save_path = self.saver.save(sess , os.path.join(self.saved_model_path, "inference", self.ckpt_name))
                print("Model saved in file: %s" % save_path)
                # ==================================================================================================

            else:
    
                start = timeit.default_timer()
        
                loss = 0
                acc = 0
        
                curr_idx = 0
                iteration = 0
                while True:
                    
                    try:
                        
                        if curr_idx >= len(self.test_dataset[0]):
                            break
                        
                        next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                        next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                        
                        if len(next_x_images) < self.batch_size:
                            break
                        
                        curr_idx = curr_idx + self.batch_size
                        iteration = iteration + 1
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                    fd_test = {
                                self.input: next_x_images, 
                                self.label: next_test_y, 
                                self.dropout_rate: 0,
                              }
    
                    curr_loss, curr_acc, normal_img, ano_img, feature_img = sess.run([self.loss, self.acc, self.normal_sample, self.ano_sample, self.feature_map], feed_dict=fd_test)               
    
                    print(normal_img.shape)
                    print(ano_img.shape)
                    
                    for i in range(len(normal_img)):
                        imageio.imwrite("./output/normal/{}_{}.png".format(curr_idx, i), np.squeeze((normal_img[i]*255).astype(np.uint8)))
    
                    for i in range(len(ano_img)):
                        imageio.imwrite("./output/abnormal/{}_{}.png".format(curr_idx, i), np.squeeze((ano_img[i]*255).astype(np.uint8)))
    
                    for i in range(len(feature_img)):
                        imageio.imwrite("./output/feature/{}_{}.png".format(curr_idx, i), np.squeeze((feature_img[i])))
                    
                    loss = loss + curr_loss
                    acc = acc + curr_acc
                
                stop = timeit.default_timer()
                
                loss = loss / iteration
                acc = acc / iteration
                
                print("Test loss: {}".format(loss))
                print("Test acc.: {}".format(acc))
                print("Time: {}".format(stop-start))

    def build_ResNet10(self):
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Classifier =============================================================================================================
        logits, _ = mz.build_model({"input":self.input, "reuse":False})  

        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)
                                   
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label)) + self.reg_set_l2_loss
        self.pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        correct_pred = tf.equal(self.pred, tf.argmax(self.label, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))             

        train_variables = tf.trainable_variables()       
        var = [v for v in train_variables if v.name.startswith(("ResNet10"))]

        self.train_var = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss, var_list=var)

        slim.model_analyzer.analyze_vars(train_variables, print_info=True)

        with tf.name_scope('train_summary'):

            tf.summary.scalar("loss", self.loss, collections=['train'])           
            tf.summary.scalar("acc", self.acc, collections=['train'])           
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("loss", self.loss, collections=['test'])           
            tf.summary.scalar("acc", self.acc, collections=['test'])           
            
            self.merged_summary_test = tf.summary.merge_all('test')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_ResNet10(self):

        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        logits, conv_1 = mz.build_model({"input":self.input, "reuse":False})  

        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label)) + self.reg_set_l2_loss
        self.pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        correct_pred = tf.equal(self.pred, tf.argmax(self.label, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 

        self.normal_sample = tf.gather(self.input, tf.where(tf.equal(self.pred, 0)))
        self.ano_sample = tf.gather(self.input, tf.where(tf.equal(self.pred, 1)))
        
        self.feature_map = tf.reduce_mean(conv_1, axis=-1, keep_dims=True)

        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        
        self.saver = tf.train.Saver()

    def train_ResNet10(self):
        
        new_learning_rate = self.learn_rate_init
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            while step <= self.max_iters:
                
                # Get the training batch
                next_x, next_y = get_batch(self.dataset, self.batch_size)
                   
                fd = {
                        self.input: next_x, 
                        self.label: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                        self.is_training: True
                     }
                
                sess.run(self.train_var, feed_dict=fd) 

                # Update Learning rate                
                if step == 2000 or step == 4000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%100 == 0:

                    # Training set
                    train_sum, train_loss, train_acc = sess.run([self.merged_summary_train, self.loss, self.acc], feed_dict=fd)
                    
                    next_valid_x, next_valid_y = get_batch(self.valid_dataset, self.batch_size)

                    fd_test = {
                                self.input: next_valid_x, 
                                self.label: next_valid_y, 
                                self.dropout_rate: 0,
                                self.is_training: False
                              }
                                           
                    test_sum, test_loss, test_acc = sess.run([self.merged_summary_test, self.loss, self.acc], feed_dict=fd_test)  
                                          
                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Train acc = [%.7f], Test loss = [%.7f], Test acc = [%.7f]" % (step, new_learning_rate, train_loss, train_acc, test_loss, test_acc))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   

                    if abs(best_loss) > abs(test_loss):
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("\n* Save ckpt: {}, Test loss: {}\n".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step, 100) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print("Model saved in file: %s" % save_path)
            print("Best loss: {}".format(best_loss))

    def test_ResNet10(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
 
            if self.output_inf_model == True:
            
                # Save inference model (batch_size=1)===============================================================
                fd_test = {
                            self.input: self.test_dataset[0][0:1], 
                            self.label: self.test_dataset[1][0:1], 
                            self.dropout_rate: 0,
                            self.is_training: False
                          }
    
                curr_loss, curr_acc = sess.run([self.loss, self.acc], feed_dict=fd_test)               
    
                mkdir_p(os.path.join(self.saved_model_path, "inference"))
                save_path = self.saver.save(sess , os.path.join(self.saved_model_path, "inference", self.ckpt_name))
                print("Model saved in file: %s" % save_path)
                # ==================================================================================================

            else:
    
                start = timeit.default_timer()
        
                loss = 0
                acc = 0
        
                curr_idx = 0
                iteration = 0
                while True:
                    
                    try:
                        
                        if curr_idx >= len(self.test_dataset[0]):
                            break
                        
                        next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                        next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                        
                        if len(next_x_images) < self.batch_size:
                            break
                        
                        curr_idx = curr_idx + self.batch_size
                        iteration = iteration + 1
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                    fd_test = {
                                self.input: next_x_images, 
                                self.label: next_test_y, 
                                self.dropout_rate: 0,
                              }
    
                    curr_loss, curr_acc, normal_img, ano_img, feature_img = sess.run([self.loss, self.acc, self.normal_sample, self.ano_sample, self.feature_map], feed_dict=fd_test)               
    
                    print(normal_img.shape)
                    print(ano_img.shape)
                    
                    for i in range(len(normal_img)):
                        imageio.imwrite("./output/normal/{}_{}.png".format(curr_idx, i), np.squeeze((normal_img[i]*255).astype(np.uint8)))
    
                    for i in range(len(ano_img)):
                        imageio.imwrite("./output/abnormal/{}_{}.png".format(curr_idx, i), np.squeeze((ano_img[i]*255).astype(np.uint8)))
    
                    for i in range(len(feature_img)):
                        imageio.imwrite("./output/feature/{}_{}.png".format(curr_idx, i), np.squeeze((feature_img[i])))
                    
                    loss = loss + curr_loss
                    acc = acc + curr_acc
                
                stop = timeit.default_timer()
                
                loss = loss / iteration
                acc = acc / iteration
                
                print("Test loss: {}".format(loss))
                print("Test acc.: {}".format(acc))
                print("Time: {}".format(stop-start))

    def build_EXAMPLE_CNN(self):
        
        # Replace the default input shape
        self.input = tf.placeholder(tf.float32, [self.batch_size, 28, 28, 1], name='input')
        self.label = tf.placeholder(tf.float32, [self.batch_size, 10], name='label')
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        ### Build model       
        # Classifier =============================================================================================================
        logits = mz.build_model({"input":self.input, "reuse":False})  

        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)
                                   
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label)) + self.reg_set_l2_loss
        self.pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        correct_pred = tf.equal(self.pred, tf.argmax(self.label, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))             

        train_variables = tf.trainable_variables()       
        var = [v for v in train_variables if v.name.startswith(("CNN"))]

        self.train_var = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.loss, var_list=var)
        
        with tf.name_scope('train_summary'):

            tf.summary.scalar("loss", self.loss, collections=['train'])           
            tf.summary.scalar("acc", self.acc, collections=['train'])           
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("loss", self.loss, collections=['test'])           
            tf.summary.scalar("acc", self.acc, collections=['test'])           
            
            self.merged_summary_test = tf.summary.merge_all('test')          
        
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver() 

    def build_eval_EXAMPLE_CNN(self):

        # Replace the default input shape
        self.input = tf.placeholder(tf.float32, [self.batch_size, 28, 28, 1], name='input')
        self.label = tf.placeholder(tf.float32, [self.batch_size, 10], name='label')
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.input, dropout=self.dropout, is_training=self.is_training, model_ticket=self.model_ticket)        
        
        logits = mz.build_model({"input":self.input, "reuse":False})  

        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        
        self.reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_set_l2_loss = tf.add_n(self.reg_set)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label)) + self.reg_set_l2_loss
        self.pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        correct_pred = tf.equal(self.pred, tf.argmax(self.label, axis=1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 

        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        
        self.saver = tf.train.Saver()

    def train_EXAMPLE_CNN(self):
        
        new_learning_rate = self.learn_rate_init
              
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        best_loss = 1000

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            mnist = input_data.read_data_sets("sdc1/dataset/MNIST_data/", one_hot = True)
            x_train = mnist.train.images
            y_train = mnist.train.labels
            x_test = mnist.test.images
            y_test = mnist.test.labels
            
            x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
            x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))
            
            print(x_train.shape)
            print(y_train.shape)
            
            train_data = (x_train, y_train)
            test_data = (x_test, y_test)

            if self.restore_model == True:
                print("Restore model: {}".format(self.train_ckpt))
                self.saver.restore(sess, self.train_ckpt)
                step = self.restore_step                
                
            else:
                step = 0

            while step <= self.max_iters:
                
                # Get the training batch
                next_x, next_y = get_batch(train_data, self.batch_size)
                   
                fd = {
                        self.input: next_x, 
                        self.label: next_y, 
                        self.dropout_rate: self.dropout,
                        self.lr: new_learning_rate,
                        self.is_training: True
                     }
                
                sess.run(self.train_var, feed_dict=fd) 

                # Update Learning rate                
                if step == 2000 or step == 4000:
                    new_learning_rate = new_learning_rate * 0.1
                    print("STEP {}, Learning rate: {}".format(step, new_learning_rate))
                
                # Record
                if step%100 == 0:

                    # Training set
                    train_sum, train_loss, train_acc = sess.run([self.merged_summary_train, self.loss, self.acc], feed_dict=fd)
                    
                    next_valid_x, next_valid_y = get_batch(test_data, self.batch_size)

                    fd_test = {
                                self.input: next_valid_x, 
                                self.label: next_valid_y, 
                                self.dropout_rate: 0,
                                self.is_training: False
                              }
                                           
                    test_sum, test_loss, test_acc = sess.run([self.merged_summary_test, self.loss, self.acc], feed_dict=fd_test)  
                                          
                    print("Step %d: LR = [%.7f], Train loss = [%.7f], Train acc = [%.7f], Test loss = [%.7f], Test acc = [%.7f]" % (step, new_learning_rate, train_loss, train_acc, test_loss, test_acc))
                    
                    summary_writer.add_summary(train_sum, step)                   
                    summary_writer.add_summary(test_sum, step)                   

                    if abs(best_loss) > abs(test_loss):
                        
                        best_loss = test_loss
                        
                        ckpt_path = os.path.join(self.saved_model_path, 'best_performance', self.ckpt_name + '_%.4f' % (best_loss))
                        print("* Save ckpt: {}, Test loss: {}".format(ckpt_path, best_loss))
                        self.best_saver.save(sess, ckpt_path, global_step=step)

                if np.mod(step, 100) == 0 and step != 0:

                    self.saver.save(sess, os.path.join(self.saved_model_path, self.ckpt_name), global_step=step)

                step += 1

            print("Best loss: {}".format(best_loss))

    def test_EXAMPLE_CNN(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            print("Restore model: {}".format(self.test_ckpt))
            self.saver.restore(sess, self.test_ckpt)            
 
            mnist = input_data.read_data_sets("sdc1/dataset/MNIST_data/", one_hot = True)
            
            x_test = mnist.test.images
            y_test = mnist.test.labels
            x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))
            
            self.test_dataset = (x_test, y_test)
            
            if self.output_inf_model == True:
                
                # Save inference model (batch_size=1)===============================================================
                fd_test = {
                            self.input: self.test_dataset[0][0:1], 
                            self.label: self.test_dataset[1][0:1], 
                            self.dropout_rate: 0,
                            self.is_training: False
                          }
    
                curr_loss, curr_acc = sess.run([self.loss, self.acc], feed_dict=fd_test)               
    
                mkdir_p(os.path.join(self.saved_model_path, "inference"))
                save_path = self.saver.save(sess , os.path.join(self.saved_model_path, "inference", self.ckpt_name))
                print("Model saved in file: %s" % save_path)
                # ==================================================================================================
                
            else:
                
                start = timeit.default_timer()
        
                loss = 0
                acc = 0
        
                curr_idx = 0
                iteration = 0
                while True:
                    
                    try:
                        
                        if curr_idx >= len(self.test_dataset[0]):
                            break
                        
                        next_x_images = self.test_dataset[0][curr_idx:curr_idx+self.batch_size]
                        next_test_y = self.test_dataset[1][curr_idx:curr_idx+self.batch_size]
                        
                        if len(next_x_images) < self.batch_size:
                            break
                        
                        curr_idx = curr_idx + self.batch_size
                        iteration = iteration + 1
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                    fd_test = {
                                self.input: next_x_images, 
                                self.label: next_test_y, 
                                self.dropout_rate: 0,
                              }
    
                    curr_loss, curr_acc = sess.run([self.loss, self.acc], feed_dict=fd_test)               
                    
                    loss = loss + curr_loss
                    acc = acc + curr_acc
                
                stop = timeit.default_timer()
                
                loss = loss / iteration
                acc = acc / iteration
                            
                print("Test loss: {}".format(loss))
                print("Test acc.: {}".format(acc))
                print("Time: {}".format(stop-start))









