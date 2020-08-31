from uuid import getnode as get_mac

class config:

    def __init__(self, configuration):
		
        self.configuration = configuration
        self.config = {
                        "common":{},
        						"train":{},
                        "test":{},
                        }

        self.mac = get_mac()
        if self.mac == 189250941727334:
            self.default_path = "/data/wei/"
        elif self.mac == 229044592702658:
            self.default_path = "/home/sdc1/"
        else:
            self.default_path = "D:/ML/"

        self.get_config()

    def get_config(self):

        try:
            conf = getattr(self, self.configuration)
            conf()

        except: 
            print("Can not find configuration")
            raise
  
    def Default_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 256
        common_config["max_iters"] = 10000
        common_config["learn_rate_init"] = 0.005
        common_config["dropout"] = 0.25

        #common_config["model_ticket"] = "CNN_v1"        
        common_config["model_ticket"] = "ResNet10"     
   
        #common_config["ckpt_name"] = "CNN_v1"
        #common_config["ckpt_name"] = "CNN_v1_AUG"
        #common_config["ckpt_name"] = "CNN_v2_AUG"
        #common_config["ckpt_name"] = "CNN_v2_AUG_REG"
        #common_config["ckpt_name"] = "CNN_v2_AUG_REG_64BCH"
        #common_config["ckpt_name"] = "CNN_v2_AUG_REG_64BCH_MP"
        #common_config["ckpt_name"] = "CNN_v2_AUG_REG_64BCH_MP_newdata"
        #common_config["ckpt_name"] = "CNN_v2_AUG_REG_256BCH_MP_newdata_drop_75_lr_0005"
        #common_config["ckpt_name"] = "CNN_v2_AUG_REG_256BCH_MP_newdata_drop_25_lr_0005"
        #common_config["ckpt_name"] = "ResNet10_AUG_256BCH_lr_001"
        common_config["ckpt_name"] = "ResNet10_AUG_REG_256BCH_lr_001"
        #common_config["ckpt_name"] = "ResNet10_AUG_REG_256BCH_lr_0001"
        #common_config["ckpt_name"] = "ResNet10_AUG_REG_256BCH_lr_005"
        
        common_config["train_data_path"] = self.default_path + "dataset/MRT/preprocessed_2nd/preprocess_train.p"
        common_config["valid_data_path"] = self.default_path + "dataset/MRT/preprocessed_2nd/preprocess_test.p"
        common_config["anomaly_data_path"] = self.default_path + "dataset/MRT/preprocessed_2nd/preprocess_test.p"
        common_config["test_data_path"] = [self.default_path + "dataset/MRT/preprocessed_2nd/preprocess_test.p"]
        
        common_config["ckpt_dir"] = self.default_path + "model/MRT/MRT-TP/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = self.default_path + "model/MRT/MRT-TP/CNN_v2_AUG_REG_64BCH_MP/best_performance/CNN_v2_AUG_REG_64BCH_MP_0.0427-9900"                
        common_config["train_ckpt"] = self.default_path + "model/MRT/MRT-TP/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls-2000"
        common_config["log_dir"] = self.default_path + "model/MRT/MRT-TP/log/" + common_config["ckpt_name"]                          
        common_config["is_training"] = True
        #common_config["is_training"] = False
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 0

        common_config["output_inf_model"] = False

    def Default_inference_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 64
        common_config["max_iters"] = 10000
        common_config["learn_rate_init"] = 0.001
        common_config["dropout"] = 0.5

        #common_config["model_ticket"] = "CNN_v1"      
        common_config["model_ticket"] = "ResNet10"      

        #common_config["ckpt_name"] = "CNN_v2_AUG_REG_64BCH_MP"
        common_config["ckpt_name"] = "ResNet10_AUG_REG_256BCH_lr_001"
        
        common_config["train_data_path"] = self.default_path + "dataset/MRT/preprocessed_2nd/preprocess_train.p"
        common_config["valid_data_path"] = self.default_path + "dataset/MRT/preprocessed_2nd/preprocess_test.p"
        common_config["anomaly_data_path"] = self.default_path + "dataset/MRT/preprocessed_2nd/preprocess_test.p"
        common_config["test_data_path"] = self.default_path + "dataset/MRT/preprocessed_2nd/preprocess_test.p"
        
        common_config["ckpt_dir"] = self.default_path + "model/MRT/MRT-TP/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = self.default_path + "model/MRT/MRT-TP/ResNet10_AUG_REG_256BCH_lr_001/best_performance/ResNet10_AUG_REG_256BCH_lr_001_0.3602-900"                
        common_config["train_ckpt"] = self.default_path + "model/MRT/MRT-TP/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls-2000"
        common_config["log_dir"] = self.default_path + "model/MRT/MRT-TP/log/" + common_config["ckpt_name"]                          
        common_config["is_training"] = False
        
        common_config["restore_model"] = True
        common_config["restore_step"] = 0
        
        common_config["output_inf_model"] = True

    def Default_1st_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 256
        common_config["max_iters"] = 10000
        common_config["learn_rate_init"] = 0.001
        common_config["dropout"] = 0.5

        #common_config["model_ticket"] = "CNN_1st_v1"        
        common_config["model_ticket"] = "ResNet10_half"    
   
        #common_config["ckpt_name"] = "CNN_1st_v1"
        #common_config["ckpt_name"] = "CNN_1st_v1_newdata"
        #common_config["ckpt_name"] = "CNN_1st_v1_temp"
        common_config["ckpt_name"] = "ResNet10_half_AUG_REG_256BCH_lr_001"
        
        common_config["train_data_path"] = self.default_path + "dataset/MRT/preprocessed_1st/preprocess_train.p"
        common_config["valid_data_path"] = self.default_path + "dataset/MRT/preprocessed_1st/preprocess_test.p"
        common_config["anomaly_data_path"] = self.default_path + "dataset/MRT/preprocessed_1st/preprocess_test.p"
        common_config["test_data_path"] = self.default_path + "dataset/MRT/preprocessed_1st/preprocess_test.p"
        
        common_config["ckpt_dir"] = self.default_path + "model/MRT/MRT-TP/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = self.default_path + "model/MRT/MRT-TP/CNN_1st_v1/best_performance/CNN_1st_v1_0.0359-8600"                
        common_config["train_ckpt"] = self.default_path + "model/MRT/MRT-TP/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls-2000"
        common_config["log_dir"] = self.default_path + "model/MRT/MRT-TP/log/" + common_config["ckpt_name"]                          
        common_config["is_training"] = True
        #common_config["is_training"] = False
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 0
        
        common_config["output_inf_model"] = False

    def Default_1st_inference_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 256
        common_config["max_iters"] = 10000
        common_config["learn_rate_init"] = 0.001
        common_config["dropout"] = 0.5

        #common_config["model_ticket"] = "CNN_1st_v1"        
        common_config["model_ticket"] = "ResNet10_half"    
        
        #common_config["ckpt_name"] = "CNN_1st_v1"
        common_config["ckpt_name"] = "ResNet10_half_AUG_REG_256BCH_lr_001"

        common_config["train_data_path"] = self.default_path + "dataset/MRT/preprocessed_1st/preprocess_train.p"
        common_config["valid_data_path"] = self.default_path + "dataset/MRT/preprocessed_1st/preprocess_test.p"
        common_config["anomaly_data_path"] = self.default_path + "dataset/MRT/preprocessed_1st/preprocess_test.p"
        common_config["test_data_path"] = self.default_path + "dataset/MRT/preprocessed_1st/preprocess_test.p"
        
        common_config["ckpt_dir"] = self.default_path + "model/MRT/MRT-TP/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = self.default_path + "model/MRT/MRT-TP/ResNet10_half_AUG_REG_256BCH_lr_001/best_performance/ResNet10_half_AUG_REG_256BCH_lr_001_0.0415-1200"                
        common_config["train_ckpt"] = self.default_path + "model/MRT/MRT-TP/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls-2000"
        common_config["log_dir"] = self.default_path + "model/MRT/MRT-TP/log/" + common_config["ckpt_name"]                          
        common_config["is_training"] = False
        
        common_config["restore_model"] = True
        common_config["restore_step"] = 0
        
        common_config["output_inf_model"] = True
        
    def Example_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 64
        common_config["max_iters"] = 10000
        common_config["learn_rate_init"] = 0.001
        common_config["dropout"] = 0.5

        common_config["model_ticket"] = "EXAMPLE_CNN"        

        common_config["ckpt_name"] = "EXAMPLE_CNN_MNIST"
        
        common_config["train_data_path"] = None
        common_config["valid_data_path"] = None
        common_config["anomaly_data_path"] = None
        common_config["test_data_path"] = None
        
        common_config["ckpt_dir"] = self.default_path + "model/MRT/MRT-TP/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = self.default_path + "model/MRT/MRT-TP/CNN_v2_AUG_REG_64BCH_MP/best_performance/CNN_v2_AUG_REG_64BCH_MP_0.0427-9900"                
        common_config["train_ckpt"] = self.default_path + "model/MRT/MRT-TP/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls-2000"
        common_config["log_dir"] = self.default_path + "model/MRT/MRT-TP/log/" + common_config["ckpt_name"]                          
        common_config["is_training"] = True
        #common_config["is_training"] = False
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 0        
        
        common_config["output_inf_model"] = False
        
    def Example_inference_config(self):
        
        # Train config 
        common_config = self.config["common"]        
        common_config["batch_size"] = 64
        common_config["max_iters"] = 10000
        common_config["learn_rate_init"] = 0.001
        common_config["dropout"] = 0

        common_config["model_ticket"] = "EXAMPLE_CNN"        

        common_config["ckpt_name"] = "EXAMPLE_CNN_MNIST"
        
        common_config["train_data_path"] = None
        common_config["valid_data_path"] = None
        common_config["anomaly_data_path"] = None
        common_config["test_data_path"] = None
        
        common_config["ckpt_dir"] = self.default_path + "model/MRT/MRT-TP/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = self.default_path + "model/MRT/MRT-TP/EXAMPLE_CNN_MNIST/EXAMPLE_CNN_MNIST-9900"                
        common_config["train_ckpt"] = self.default_path + "model/MRT/MRT-TP/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls-2000"
        common_config["log_dir"] = self.default_path + "model/MRT/MRT-TP/log/" + common_config["ckpt_name"]                          
        common_config["is_training"] = False
        
        common_config["restore_model"] = True
        common_config["restore_step"] = 0    

        common_config["output_inf_model"] = True        

