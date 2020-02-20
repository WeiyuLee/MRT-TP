import tensorflow as tf

import os
os.environ['KERAS_BACKEND']='tensorflow'

import sys
sys.path.append('./utility')

from utils import mkdir_p
from model import model
from utils import InputData
import config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="Default_config", help="Configuration name")
args = parser.parse_args()
conf = config.config(args.config).config["common"]

#FLAGS = flags.FLAGS
if __name__ == "__main__":

    root_log_dir = conf["log_dir"]
    checkpoint_dir = conf["ckpt_dir"]
       
    mkdir_p(root_log_dir)
    mkdir_p(checkpoint_dir)
    mkdir_p(os.path.join(checkpoint_dir, 'best_performance'))

    model_path = checkpoint_dir

    max_iters = conf["max_iters"]
    dropout = conf["dropout"]
    ckpt_name = conf["ckpt_name"]
    test_ckpt = conf["test_ckpt"]
    train_ckpt = conf["train_ckpt"]
    restore_model = conf["restore_model"]
    restore_step = conf["restore_step"]
    model_ticket = conf["model_ticket"]
    
    is_training = conf["is_training"]
    
    learn_rate_init = conf["learn_rate_init"]
    
    output_inf_model = conf["output_inf_model"]
    
    print("===================================================================")
    if is_training == True:
        batch_size = conf["batch_size"]
        print("*** [Training] ***")
        print("restore_model: [{}]".format(restore_model))
        print("train_data_path: [{}]".format(conf["train_data_path"]))
        print("valid_data_path: [{}]".format(conf["valid_data_path"]))        
        print("anomaly_data_path: [{}]".format(conf["anomaly_data_path"]))        
        print("ckpt_name: [{}]".format(ckpt_name))
        print("max_iters: [{}]".format(max_iters))   
        print("learn_rate_init: [{}]".format(learn_rate_init))
    else:
        if output_inf_model == True:
            batch_size = 1
            print("*** [Output Inference Model] ***")
        else:
            batch_size = 64
            print("*** [Testing] ***")
        print("test_data_path: [{}]".format(conf["test_data_path"]))

    print("batch_size: [{}]".format(batch_size))   
    print("model_ticket: [{}]".format(model_ticket))   
    print("dropout: [{}]".format(dropout))
    print("===================================================================")
    
    if is_training == True:               
        
        cb_ob = InputData(conf["train_data_path"], conf["valid_data_path"], conf["anomaly_data_path"], None)

        MODEL = model(  batch_size=batch_size, 
                        max_iters=max_iters, 
                        dropout=dropout,
                        model_path=model_path, 
                        data_ob=cb_ob, 
                        log_dir=root_log_dir, 
                        learnrate_init=learn_rate_init,
                        ckpt_name=ckpt_name,
                        test_ckpt=test_ckpt,
                        train_ckpt=train_ckpt,
                        restore_model=restore_model,
                        restore_step=restore_step,
                        model_ticket=model_ticket,
                        is_training=is_training,
                        output_inf_model=output_inf_model)
        
        MODEL.build_model()
        MODEL.train()

    else:

        test_data_path = conf["test_data_path"]            
        
        cb_ob = InputData(None, None, None, test_data_path)

        MODEL = model(  batch_size=batch_size, 
                        max_iters=max_iters, 
                        dropout=dropout,
                        model_path=model_path, 
                        data_ob=cb_ob, 
                        log_dir=root_log_dir, 
                        learnrate_init=learn_rate_init,
                        ckpt_name=ckpt_name,
                        test_ckpt=test_ckpt,
                        train_ckpt=train_ckpt,
                        restore_model=restore_model,
                        restore_step=restore_step,
                        model_ticket=model_ticket,
                        is_training=is_training,
                        output_inf_model=output_inf_model)
        
        MODEL.build_eval_model()
        MODEL.test()









