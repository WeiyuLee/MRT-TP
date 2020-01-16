import tensorflow as tf
import netfactory as nf

class model_zoo:
    
    def __init__(self, inputs, dropout, is_training, model_ticket):
        
        self.model_ticket = model_ticket
        self.inputs = inputs
        self.dropout = dropout
        self.is_training = is_training

    def build_model(self, kwargs = {}):

        model_list = ["CNN_v1", "EXAMPLE_CNN"]
        
        if self.model_ticket not in model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
           
            fn = getattr(self,self.model_ticket)
            
            if kwargs == {}:
                netowrk = fn()
            else:
                netowrk = fn(kwargs)
            return netowrk
        
    def CNN_v1(self, kwargs):
        
            model_params = {       
    
                "conv_1": [11,11,128],
                "conv_2": [5,5,256],
                "conv_3": [3,3,512],
                "conv_4": [3,3,1024],
                "conv_5": [3,3,1024],
                
                "fc_1": 1024,
                "fc_2": 512,
                "fc_out": 2,
                
            }
    
            reuse = kwargs["reuse"]
            l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   

            print("===================================================================")
            
            with tf.variable_scope("CNN", reuse=reuse):
                
                input = kwargs["input"]
                
                print("[CNN_v1] input: %s" % input.get_shape())
            
#                conv_1_1 = nf.convolution_layer(input, model_params["conv_1"], [1,4,4,1], name="conv_1_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
#                conv_1_2 = nf.convolution_layer(conv_1_1, model_params["conv_1"], [1,1,1,1], name="conv_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
#                conv_1 = conv_1_1 + conv_1_2
#                conv_1 = tf.layers.dropout(conv_1, rate=self.dropout, training=self.is_training, name='conv_1_dropout')
#                print("conv_1: %s" % conv_1.get_shape())     
#                
#                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,3,3,1], name="conv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
#                conv_2_2 = nf.convolution_layer(conv_2_1, model_params["conv_2"], [1,1,1,1], name="conv_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
#                conv_2 = conv_2_1 + conv_2_2
#                conv_2 = tf.layers.dropout(conv_2, rate=self.dropout, training=self.is_training, name='conv_2_dropout')
#                print("conv_2: %s" % conv_2.get_shape())     
#
#                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,3,3,1], name="conv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
#                conv_3_2 = nf.convolution_layer(conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
#                conv_3 = conv_3_1 + conv_3_2
#                conv_3 = tf.layers.dropout(conv_3, rate=self.dropout, training=self.is_training, name='conv_3_dropout')
#                print("conv_3: %s" % conv_3.get_shape())     
#                
#                conv_4_1 = nf.convolution_layer(conv_3, model_params["conv_4"], [1,2,2,1], name="conv_4_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
#                conv_4_2 = nf.convolution_layer(conv_4_1, model_params["conv_4"], [1,1,1,1], name="conv_4_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
#                conv_4 = conv_4_1 + conv_4_2
#                conv_4 = tf.layers.dropout(conv_4, rate=self.dropout, training=self.is_training, name='conv_4_dropout')
#                print("conv_4: %s" % conv_4.get_shape()) 
#                
#                conv_5_1 = nf.convolution_layer(conv_4, model_params["conv_5"], [1,2,2,1], name="conv_5_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
#                conv_5_2 = nf.convolution_layer(conv_5_1, model_params["conv_5"], [1,1,1,1], name="conv_5_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
#                conv_5 = conv_5_1 + conv_5_2
#                conv_5 = tf.layers.dropout(conv_5, rate=self.dropout, training=self.is_training, name='conv_5_dropout')                
#                print("conv_5: %s" % conv_5.get_shape()) 
#                
#                conv_code = tf.reshape(conv_5, [tf.shape(self.inputs)[0], 3*2*1024])                   
#                fc_1 = nf.fc_layer(conv_code, model_params["fc_1"], name="fc_1", activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)
#                fc_1 = tf.layers.dropout(fc_1, rate=self.dropout, training=self.is_training, name='fc_1_dropout')
#                print("fc_1: %s" % fc_1.get_shape())     
#                
#                fc_out = nf.fc_layer(fc_1, model_params["fc_out"], name="fc_out", activat_fn=None)

                conv_1_1 = nf.convolution_layer(input, model_params["conv_1"], [1,4,4,1], name="conv_1_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_1_2 = nf.convolution_layer(conv_1_1, model_params["conv_1"], [1,1,1,1], name="conv_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_1 = conv_1_1 + conv_1_2
                conv_1 = tf.nn.max_pool(conv_1, [1,3,3,1], [1,2,2,1], padding='VALID')
                conv_1 = tf.layers.dropout(conv_1, rate=self.dropout, training=self.is_training, name='conv_1_dropout')
                print("conv_1: %s" % conv_1.get_shape())     
                
                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_2_2 = nf.convolution_layer(conv_2_1, model_params["conv_2"], [1,1,1,1], name="conv_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_2 = conv_2_1 + conv_2_2
                conv_2 = tf.nn.max_pool(conv_2, [1,3,3,1], [1,2,2,1], padding='VALID')
                conv_2 = tf.layers.dropout(conv_2, rate=self.dropout, training=self.is_training, name='conv_2_dropout')
                print("conv_2: %s" % conv_2.get_shape())     

                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,1,1,1], name="conv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_3_2 = nf.convolution_layer(conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_3 = conv_3_1 + conv_3_2
                conv_3 = tf.layers.dropout(conv_3, rate=self.dropout, training=self.is_training, name='conv_3_dropout')
                print("conv_3: %s" % conv_3.get_shape())     
                
                conv_4_1 = nf.convolution_layer(conv_3, model_params["conv_4"], [1,1,1,1], name="conv_4_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_4_2 = nf.convolution_layer(conv_4_1, model_params["conv_4"], [1,1,1,1], name="conv_4_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_4 = conv_4_1 + conv_4_2
                conv_4 = tf.layers.dropout(conv_4, rate=self.dropout, training=self.is_training, name='conv_4_dropout')
                print("conv_4: %s" % conv_4.get_shape()) 
                
                conv_5_1 = nf.convolution_layer(conv_4, model_params["conv_5"], [1,1,1,1], name="conv_5_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_5_2 = nf.convolution_layer(conv_5_1, model_params["conv_5"], [1,1,1,1], name="conv_5_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_5 = conv_5_1 + conv_5_2
                conv_5 = tf.nn.max_pool(conv_5, [1,3,3,1], [1,2,2,1], padding='VALID')
                conv_5 = tf.layers.dropout(conv_5, rate=self.dropout, training=self.is_training, name='conv_5_dropout')                
                print("conv_5: %s" % conv_5.get_shape()) 
                
                conv_code = tf.reshape(conv_5, [tf.shape(self.inputs)[0], 4*2*1024])                   
                fc_1 = nf.fc_layer(conv_code, model_params["fc_1"], name="fc_1", activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)
                fc_1 = tf.layers.dropout(fc_1, rate=self.dropout, training=self.is_training, name='fc_1_dropout')
                print("fc_1: %s" % fc_1.get_shape())     

                fc_2 = nf.fc_layer(fc_1, model_params["fc_2"], name="fc_2", activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)
                fc_2 = tf.layers.dropout(fc_2, rate=self.dropout, training=self.is_training, name='fc_2_dropout')
                print("fc_2: %s" % fc_2.get_shape())     
                
                fc_out = nf.fc_layer(fc_2, model_params["fc_out"], name="fc_out", activat_fn=None)
                print("fc_out: %s" % fc_out.get_shape())  
                
                return fc_out, conv_1

    def EXAMPLE_CNN(self, kwargs):
        
            model_params = {       
    
                "conv_1": [3,3,128],
                "conv_2": [3,3,256],
                
                "fc_1": 1024,
                "fc_2": 512,
                "fc_out": 10,
                
            }
    
            reuse = kwargs["reuse"]
            l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   

            print("===================================================================")
            
            with tf.variable_scope("CNN", reuse=reuse):
                
                input = kwargs["input"]
                
                print("[EXAMPLE_CNN] input: %s" % input.get_shape())

                conv_1_1 = nf.convolution_layer(input, model_params["conv_1"], [1,1,1,1], name="conv_1_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_1_2 = nf.convolution_layer(conv_1_1, model_params["conv_1"], [1,1,1,1], name="conv_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_1 = conv_1_1 + conv_1_2
                conv_1 = tf.nn.max_pool(conv_1, [1,2,2,1], [1,2,2,1], padding='VALID')
                conv_1 = tf.layers.dropout(conv_1, rate=self.dropout, training=self.is_training, name='conv_1_dropout')
                print("conv_1: %s" % conv_1.get_shape())     
                
                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,1,1,1], name="conv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_2_2 = nf.convolution_layer(conv_2_1, model_params["conv_2"], [1,1,1,1], name="conv_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                conv_2 = conv_2_1 + conv_2_2
                conv_2 = tf.nn.max_pool(conv_2, [1,2,2,1], [1,2,2,1], padding='VALID')
                conv_2 = tf.layers.dropout(conv_2, rate=self.dropout, training=self.is_training, name='conv_2_dropout')
                print("conv_2: %s" % conv_2.get_shape())     
                
                conv_code = tf.reshape(conv_2, [tf.shape(self.inputs)[0], 7*7*256])                   
                fc_1 = nf.fc_layer(conv_code, model_params["fc_1"], name="fc_1", activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)
                fc_1 = tf.layers.dropout(fc_1, rate=self.dropout, training=self.is_training, name='fc_1_dropout')
                print("fc_1: %s" % fc_1.get_shape())     

                fc_2 = nf.fc_layer(fc_1, model_params["fc_2"], name="fc_2", activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)
                fc_2 = tf.layers.dropout(fc_2, rate=self.dropout, training=self.is_training, name='fc_2_dropout')
                print("fc_2: %s" % fc_2.get_shape())     
                
                fc_out = nf.fc_layer(fc_2, model_params["fc_out"], name="fc_out", activat_fn=None)
                print("fc_out: %s" % fc_out.get_shape())  
                
                return fc_out
            
def unit_test():

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
    is_training = tf.placeholder(tf.bool, name='is_training')
    dropout = tf.placeholder(tf.float32, name='dropout')
    
    mz = model_zoo(x, dropout, is_training,"resNet_v1")
    return mz.build_model()
