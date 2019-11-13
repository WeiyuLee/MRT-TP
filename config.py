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
        common_config["batch_size"] = 64
        common_config["max_iters"] = 10000
        common_config["learn_rate_init"] = 0.001
        common_config["dropout"] = 0.5

        common_config["model_ticket"] = "CNN_v1"        
   
        #common_config["ckpt_name"] = "CNN_v1"
        #common_config["ckpt_name"] = "CNN_v1_AUG"
        #common_config["ckpt_name"] = "CNN_v2_AUG"
        #common_config["ckpt_name"] = "CNN_v2_AUG_REG"
        #common_config["ckpt_name"] = "CNN_v2_AUG_REG_64BCH"
        common_config["ckpt_name"] = "CNN_v2_AUG_REG_64BCH_MP"
        
        common_config["train_data_path"] = self.default_path + "dataset/MRT/preprocessed/preprocess_train.p"
        common_config["valid_data_path"] = self.default_path + "dataset/MRT/preprocessed/preprocess_test.p"
        common_config["anomaly_data_path"] = self.default_path + "dataset/MRT/preprocessed/preprocess_test.p"
        common_config["test_data_path"] = [self.default_path + "dataset/MRT/preprocessed/preprocess_test.p"]
        
        common_config["ckpt_dir"] = self.default_path + "model/MRT/MRT-TP/" + common_config["ckpt_name"]      
        common_config["test_ckpt"] = self.default_path + "model/MRT/MRT-TP/CNN_v2_AUG_REG_64BCH_MP/best_performance/CNN_v2_AUG_REG_64BCH_MP_0.0427-9900"                
        common_config["train_ckpt"] = self.default_path + "model/MRT/MRT-TP/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls/AD_AE_GAN_3DCode_v1_512_R50_C50_10cls-2000"
        common_config["log_dir"] = self.default_path + "model/MRT/MRT-TP/log/" + common_config["ckpt_name"]                          
        #common_config["is_training"] = True
        common_config["is_training"] = False
        
        common_config["restore_model"] = False
        common_config["restore_step"] = 0

