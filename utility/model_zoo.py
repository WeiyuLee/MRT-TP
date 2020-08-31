import tensorflow as tf
import netfactory as nf

class model_zoo:
    
    def __init__(self, inputs, dropout, is_training, model_ticket):
        
        self.model_ticket = model_ticket
        self.inputs = inputs
        self.dropout = dropout
        self.is_training = is_training

    def build_model(self, kwargs = {}):

        model_list = ["CNN_v1", "CNN_1st_v1", "ResNet10_half", "ResNet10", "EXAMPLE_CNN"]
        
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

    def CNN_1st_v1(self, kwargs):
        
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
                
                print("[CNN_1st_v1] input: %s" % input.get_shape())

                conv_1_1 = nf.convolution_layer(input, model_params["conv_1"], [1,4,4,1], name="conv_1_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                #conv_1_2 = nf.convolution_layer(conv_1_1, model_params["conv_1"], [1,1,1,1], name="conv_1_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                #conv_1 = conv_1_1 + conv_1_2
                conv_1 = tf.nn.max_pool(conv_1_1, [1,3,3,1], [1,2,2,1], padding='VALID')
                conv_1 = tf.layers.dropout(conv_1, rate=self.dropout, training=self.is_training, name='conv_1_dropout')
                print("conv_1: %s" % conv_1.get_shape())     
                
                conv_2_1 = nf.convolution_layer(conv_1, model_params["conv_2"], [1,2,2,1], name="conv_2_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                #conv_2_2 = nf.convolution_layer(conv_2_1, model_params["conv_2"], [1,1,1,1], name="conv_2_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                #conv_2 = conv_2_1 + conv_2_2
                conv_2 = tf.nn.max_pool(conv_2_1, [1,3,3,1], [1,2,2,1], padding='VALID')
                conv_2 = tf.layers.dropout(conv_2, rate=self.dropout, training=self.is_training, name='conv_2_dropout')
                print("conv_2: %s" % conv_2.get_shape())     

                conv_3_1 = nf.convolution_layer(conv_2, model_params["conv_3"], [1,1,1,1], name="conv_3_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                #conv_3_2 = nf.convolution_layer(conv_3_1, model_params["conv_3"], [1,1,1,1], name="conv_3_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                #conv_3 = conv_3_1 + conv_3_2
                conv_3 = tf.layers.dropout(conv_3_1, rate=self.dropout, training=self.is_training, name='conv_3_dropout')
                print("conv_3: %s" % conv_3.get_shape())     
                
                #conv_4_1 = nf.convolution_layer(conv_3, model_params["conv_4"], [1,1,1,1], name="conv_4_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                #conv_4_2 = nf.convolution_layer(conv_4_1, model_params["conv_4"], [1,1,1,1], name="conv_4_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                #conv_4 = conv_4_1 + conv_4_2
                #conv_4 = tf.layers.dropout(conv_4, rate=self.dropout, training=self.is_training, name='conv_4_dropout')
                #print("conv_4: %s" % conv_4.get_shape()) 
                
                conv_5_1 = nf.convolution_layer(conv_3, model_params["conv_5"], [1,1,1,1], name="conv_5_1", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                #conv_5_2 = nf.convolution_layer(conv_5_1, model_params["conv_5"], [1,1,1,1], name="conv_5_2", padding='SAME', activat_fn=nf.lrelu, is_bn=True, is_training=self.is_training, reg=l2_reg)               
                #conv_5 = conv_5_1 + conv_5_2
                conv_5 = tf.nn.max_pool(conv_5_1, [1,3,3,1], [1,2,2,1], padding='VALID')
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
                fc_1 = nf.fc_layer(conv_code, model_params["fc_1"], name="fc_1", activat_fn=nf.lrelu, is_bn=False, is_training=self.is_training, reg=l2_reg)
                fc_1 = tf.layers.dropout(fc_1, rate=self.dropout, training=self.is_training, name='fc_1_dropout')
                print("fc_1: %s" % fc_1.get_shape())     

                fc_2 = nf.fc_layer(fc_1, model_params["fc_2"], name="fc_2", activat_fn=nf.lrelu, is_bn=False, is_training=self.is_training, reg=l2_reg)
                fc_2 = tf.layers.dropout(fc_2, rate=self.dropout, training=self.is_training, name='fc_2_dropout')
                print("fc_2: %s" % fc_2.get_shape())     
                
                fc_out = nf.fc_layer(fc_2, model_params["fc_out"], name="fc_out", activat_fn=None)
                print("fc_out: %s" % fc_out.get_shape())  
                
                return fc_out, conv_1

    def ResNet10_half(self, kwargs):
        
            model_params = {       
    
                "conv1": [7,7,32],

                "conv2_1": [3,3,32],
                "conv2_2": [3,3,32],

                "conv3_1": [3,3,64],
                "conv3_2": [3,3,64],
                "conv3_sc": [1,1,64],

                "conv4_1": [3,3,128],
                "conv4_2": [3,3,128],
                "conv4_sc": [1,1,128],

                "conv5_1": [3,3,256],
                "conv5_2": [3,3,256],
                "conv5_sc": [1,1,256],
                
                "convM_1": [3,3,256],
                "convM_2": [3,3,256],
                "fcM_3": 8,
                "fcM_4": 2,
                
            }
    
            reuse = kwargs["reuse"]
            l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   

            print("===================================================================")
            
            with tf.variable_scope("ResNet10_half", reuse=reuse):
                
                input = kwargs["input"]
                
                print("[ResNet-10] input: %s" % input.get_shape())

                # conv 1
                x =         nf.convolution_layer(input, model_params["conv1"], [1,2,2,1], name="conv1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training)
                x =         tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], padding='SAME', name='max_1')
                print("conv1: %s" % x.get_shape()) 
                conv1 = x

                # conv 2
                sc_2 = x
                x =         nf.convolution_layer(x, model_params["conv2_1"], [1,1,1,1], name="conv2_1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv2_2"], [1,1,1,1], name="conv2_2", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         tf.add(x, sc_2)
                x =         tf.nn.relu(x, name="conv2_2"+"_out")
                print("[residual_simple_block] conv2: %s" % x.get_shape()) 

                # conv 3
                sc_3 =      nf.convolution_layer(x, model_params["conv3_sc"], [1,2,2,1], name="conv3_sc", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv3_1"], [1,2,2,1], name="conv3_1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv3_2"], [1,1,1,1], name="conv3_2", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         tf.add(x, sc_3)
                x =         tf.nn.relu(x, name="conv3_2"+"_out")
                print("[residual_simple_block] conv3: %s" % x.get_shape()) 

                # conv 4
                sc_4 =      nf.convolution_layer(x, model_params["conv4_sc"], [1,2,2,1], name="conv4_sc", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv4_1"], [1,2,2,1], name="conv4_1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv4_2"], [1,1,1,1], name="conv4_2", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         tf.add(x, sc_4)
                x =         tf.nn.relu(x, name="conv4_2"+"_out")
                print("[residual_simple_block] conv4: %s" % x.get_shape()) 

                # conv 5
                sc_5 =      nf.convolution_layer(x, model_params["conv5_sc"], [1,2,2,1], name="conv5_sc", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv5_1"], [1,2,2,1], name="conv5_1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv5_2"], [1,1,1,1], name="conv5_2", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         tf.add(x, sc_5)
                x =         tf.nn.relu(x, name="conv5_2"+"_out")
                print("[residual_simple_block] conv5: %s" % x.get_shape()) 

                # module conv 1
                x =         nf.convolution_layer(x, model_params["convM_1"], [1,1,1,1], name="convM_1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training, reg=l2_reg)
                x =         tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], padding='VALID', name='max_1')
                print("[relation_module] convM_1: %s" % x.get_shape()) 

                # module conv 2
                x =         nf.convolution_layer(x, model_params["convM_2"], [1,1,1,1], name="convM_2", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training, reg=l2_reg)
                x =         tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], padding='VALID', name='max_2')
                print("[relation_module] convM_2: %s" % x.get_shape()) 

                # module fc 3
                x =         tf.reshape(x, [tf.shape(input)[0], 3*1*256])      
                x =         nf.fc_layer(x, model_params["fcM_3"], name="fcM_3", activat_fn=tf.nn.relu, is_bn=False, is_training=self.is_training, reg=l2_reg)
                print("[relation_module] fcM_3: %s" % x.get_shape()) 

                # module fc 4
                x =         nf.fc_layer(x, model_params["fcM_4"], name="fcM_4", activat_fn=None, is_bn=False, is_training=self.is_training, reg=l2_reg)
                print("[relation_module] fcM_4: %s" % x.get_shape()) 

                return x, conv1             

    def ResNet10(self, kwargs):
        
            model_params = {       
    
                "conv1": [7,7,64],

                "conv2_1": [3,3,64],
                "conv2_2": [3,3,64],

                "conv3_1": [3,3,128],
                "conv3_2": [3,3,128],
                "conv3_sc": [1,1,128],

                "conv4_1": [3,3,256],
                "conv4_2": [3,3,256],
                "conv4_sc": [1,1,256],

                "conv5_1": [3,3,512],
                "conv5_2": [3,3,512],
                "conv5_sc": [1,1,512],
                
                "convM_1": [3,3,512],
                "convM_2": [3,3,512],
                "fcM_3": 8,
                "fcM_4": 2,
                
            }
    
            reuse = kwargs["reuse"]
            l2_reg = tf.contrib.layers.l2_regularizer(1e-5)   

            print("===================================================================")
            
            with tf.variable_scope("ResNet10", reuse=reuse):
                
                input = kwargs["input"]
                
                print("[ResNet-10] input: %s" % input.get_shape())

                # conv 1
                x =         nf.convolution_layer(input, model_params["conv1"], [1,2,2,1], name="conv1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training)
                x =         tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], padding='SAME', name='max_1')
                print("conv1: %s" % x.get_shape()) 
                conv1 = x

                # conv 2
                sc_2 = x
                x =         nf.convolution_layer(x, model_params["conv2_1"], [1,1,1,1], name="conv2_1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv2_2"], [1,1,1,1], name="conv2_2", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         tf.add(x, sc_2)
                x =         tf.nn.relu(x, name="conv2_2"+"_out")
                print("[residual_simple_block] conv2: %s" % x.get_shape()) 

                # conv 3
                sc_3 =      nf.convolution_layer(x, model_params["conv3_sc"], [1,2,2,1], name="conv3_sc", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv3_1"], [1,2,2,1], name="conv3_1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv3_2"], [1,1,1,1], name="conv3_2", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         tf.add(x, sc_3)
                x =         tf.nn.relu(x, name="conv3_2"+"_out")
                print("[residual_simple_block] conv3: %s" % x.get_shape()) 

                # conv 4
                sc_4 =      nf.convolution_layer(x, model_params["conv4_sc"], [1,2,2,1], name="conv4_sc", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv4_1"], [1,2,2,1], name="conv4_1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv4_2"], [1,1,1,1], name="conv4_2", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         tf.add(x, sc_4)
                x =         tf.nn.relu(x, name="conv4_2"+"_out")
                print("[residual_simple_block] conv4: %s" % x.get_shape()) 

                # conv 5
                sc_5 =      nf.convolution_layer(x, model_params["conv5_sc"], [1,2,2,1], name="conv5_sc", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv5_1"], [1,2,2,1], name="conv5_1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training)
                x =         nf.convolution_layer(x, model_params["conv5_2"], [1,1,1,1], name="conv5_2", padding='SAME', activat_fn=None, is_bn=True, is_training=self.is_training)
                x =         tf.add(x, sc_5)
                x =         tf.nn.relu(x, name="conv5_2"+"_out")
                print("[residual_simple_block] conv5: %s" % x.get_shape()) 

                # module conv 1
                x =         nf.convolution_layer(x, model_params["convM_1"], [1,1,1,1], name="convM_1", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training, reg=l2_reg)
                x =         tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], padding='VALID', name='max_1')
                print("[relation_module] convM_1: %s" % x.get_shape()) 

                # module conv 2
                x =         nf.convolution_layer(x, model_params["convM_2"], [1,1,1,1], name="convM_2", padding='SAME', activat_fn=tf.nn.relu, is_bn=True, is_training=self.is_training, reg=l2_reg)
                x =         tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], padding='VALID', name='max_2')
                print("[relation_module] convM_2: %s" % x.get_shape()) 

                # module fc 3
                x =         tf.reshape(x, [tf.shape(input)[0], 3*1*512])      
                x =         nf.fc_layer(x, model_params["fcM_3"], name="fcM_3", activat_fn=tf.nn.relu, is_bn=False, is_training=self.is_training, reg=l2_reg)
                print("[relation_module] fcM_3: %s" % x.get_shape()) 

                # module fc 4
                x =         nf.fc_layer(x, model_params["fcM_4"], name="fcM_4", activat_fn=None, is_bn=False, is_training=self.is_training, reg=l2_reg)
                print("[relation_module] fcM_4: %s" % x.get_shape()) 

                return x, conv1               

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
