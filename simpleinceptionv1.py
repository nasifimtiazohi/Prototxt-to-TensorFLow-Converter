import tensorflow as tf
slim=tf.contrib.slim

def InceptionV1(inputs,num_classes=1000,is_training=true,reuse=true,scope=InceptionV1):
	with tf.variable_scope(scope,"Model",reuse=reuse):
		with slim.arg_scope(default_arg_scope(is_training):
			end_points= {}
			end_points['inputs']=inputs
			end_points['Conv2d_1a_7x7']=slim.conv2d(end_points['inputs'],64,[7,7],stride=2,scope='Conv2d_1a_7x7')
			end_points['MaxPool_2a_3x3']=slim.max_pool2d(end_points['Conv2d_1a_7x7'],[3,3],stride=2,scope='MaxPool_2a_3x3')
			end_points['Conv2d_2b_1x1']=slim.conv2d(end_points['MaxPool_2a_3x3'],64,[1,1],stride=1,scope='Conv2d_2b_1x1')
			end_points['Conv2d_2c_3x3']=slim.conv2d(end_points['Conv2d_2b_1x1'],192,[3,3],stride=1,scope='Conv2d_2c_3x3')
			end_points['MaxPool_3a_3x3']=slim.max_pool2d(end_points['Conv2d_2c_3x3'],[3,3],stride=2,scope='MaxPool_3a_3x3')
			with tf.variable_scope(Mixed_3b):
				with tf.variable_scope(branch_0):
					branch_0=slim.conv2d(end_points['MaxPool_3a_3x3'],64,[1,1],stride=1,scope='Conv2d_0a_1x1')
				
				with tf.variable_scope(branch_1):
					branch_1=slim.conv2d(end_points['MaxPool_3a_3x3'],96,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_1=slim.conv2d(end_points['Mixed_3b/Branch_1/Conv2d_0a_1x1'],128,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_2):
					branch_2=slim.conv2d(end_points['MaxPool_3a_3x3'],16,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_2=slim.conv2d(end_points['Mixed_3b/Branch_2/Conv2d_0a_1x1'],32,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_3):
					branch_3=slim.max_pool2d(end_points['MaxPool_3a_3x3'],[3,3],stride=1,scope='MaxPool_0a_3x3')
					branch_3=slim.conv2d(end_points['Mixed_3b/Branch_3/MaxPool_0a_3x3'],32,[1,1],stride=1,scope='Conv2d_0b_1x1')
				
				concat_point=tf.concat(axis=3,values=[branch_0,branch_1,branch_2,branch_3])
			end_points[Mixed_3b]=concat_point
			with tf.variable_scope(Mixed_3c):
				with tf.variable_scope(branch_0):
					branch_0=slim.conv2d(end_points['Mixed_3b'],128,[1,1],stride=1,scope='Conv2d_0a_1x1')
				
				with tf.variable_scope(branch_1):
					branch_1=slim.conv2d(end_points['Mixed_3b'],128,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_1=slim.conv2d(end_points['Mixed_3c/Branch_1/Conv2d_0a_1x1'],192,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_2):
					branch_2=slim.conv2d(end_points['Mixed_3b'],32,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_2=slim.conv2d(end_points['Mixed_3c/Branch_2/Conv2d_0a_1x1'],96,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_3):
					branch_3=slim.max_pool2d(end_points['Mixed_3b'],[3,3],stride=1,scope='MaxPool_0a_3x3')
					branch_3=slim.conv2d(end_points['Mixed_3c/Branch_3/MaxPool_0a_3x3'],64,[1,1],stride=1,scope='Conv2d_0b_1x1')
				
				concat_point=tf.concat(axis=3,values=[branch_0,branch_1,branch_2,branch_3])
			end_points[Mixed_3c]=concat_point
			end_points['MaxPool_4a_3x3']=slim.max_pool2d(end_points['Mixed_3c'],[3,3],stride=2,scope='MaxPool_4a_3x3')
			with tf.variable_scope(Mixed_4b):
				with tf.variable_scope(branch_0):
					branch_0=slim.conv2d(end_points['MaxPool_4a_3x3'],192,[1,1],stride=1,scope='Conv2d_0a_1x1')
				
				with tf.variable_scope(branch_1):
					branch_1=slim.conv2d(end_points['MaxPool_4a_3x3'],96,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_1=slim.conv2d(end_points['Mixed_4b/Branch_1/Conv2d_0a_1x1'],208,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_2):
					branch_2=slim.conv2d(end_points['MaxPool_4a_3x3'],16,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_2=slim.conv2d(end_points['Mixed_4b/Branch_2/Conv2d_0a_1x1'],48,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_3):
					branch_3=slim.max_pool2d(end_points['MaxPool_4a_3x3'],[3,3],stride=1,scope='MaxPool_0a_3x3')
					branch_3=slim.conv2d(end_points['Mixed_4b/Branch_3/MaxPool_0a_3x3'],64,[1,1],stride=1,scope='Conv2d_0b_1x1')
				
				concat_point=tf.concat(axis=3,values=[branch_0,branch_1,branch_2,branch_3])
			end_points[Mixed_4b]=concat_point
			with tf.variable_scope(Mixed_4c):
				with tf.variable_scope(branch_0):
					branch_0=slim.conv2d(end_points['Mixed_4b'],160,[1,1],stride=1,scope='Conv2d_0a_1x1')
				
				with tf.variable_scope(branch_1):
					branch_1=slim.conv2d(end_points['Mixed_4b'],112,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_1=slim.conv2d(end_points['Mixed_4c/Branch_1/Conv2d_0a_1x1'],224,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_2):
					branch_2=slim.conv2d(end_points['Mixed_4b'],24,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_2=slim.conv2d(end_points['Mixed_4c/Branch_2/Conv2d_0a_1x1'],64,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_3):
					branch_3=slim.max_pool2d(end_points['Mixed_4b'],[3,3],stride=1,scope='MaxPool_0a_3x3')
					branch_3=slim.conv2d(end_points['Mixed_4c/Branch_3/MaxPool_0a_3x3'],64,[1,1],stride=1,scope='Conv2d_0b_1x1')
				
				concat_point=tf.concat(axis=3,values=[branch_0,branch_1,branch_2,branch_3])
			end_points[Mixed_4c]=concat_point
			with tf.variable_scope(Mixed_4d):
				with tf.variable_scope(branch_0):
					branch_0=slim.conv2d(end_points['Mixed_4c'],128,[1,1],stride=1,scope='Conv2d_0a_1x1')
				
				with tf.variable_scope(branch_1):
					branch_1=slim.conv2d(end_points['Mixed_4c'],128,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_1=slim.conv2d(end_points['Mixed_4d/Branch_1/Conv2d_0a_1x1'],256,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_2):
					branch_2=slim.conv2d(end_points['Mixed_4c'],24,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_2=slim.conv2d(end_points['Mixed_4d/Branch_2/Conv2d_0a_1x1'],64,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_3):
					branch_3=slim.max_pool2d(end_points['Mixed_4c'],[3,3],stride=1,scope='MaxPool_0a_3x3')
					branch_3=slim.conv2d(end_points['Mixed_4d/Branch_3/MaxPool_0a_3x3'],64,[1,1],stride=1,scope='Conv2d_0b_1x1')
				
				concat_point=tf.concat(axis=3,values=[branch_0,branch_1,branch_2,branch_3])
			end_points[Mixed_4d]=concat_point
			with tf.variable_scope(Mixed_4e):
				with tf.variable_scope(branch_0):
					branch_0=slim.conv2d(end_points['Mixed_4d'],112,[1,1],stride=1,scope='Conv2d_0a_1x1')
				
				with tf.variable_scope(branch_1):
					branch_1=slim.conv2d(end_points['Mixed_4d'],144,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_1=slim.conv2d(end_points['Mixed_4e/Branch_1/Conv2d_0a_1x1'],288,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_2):
					branch_2=slim.conv2d(end_points['Mixed_4d'],32,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_2=slim.conv2d(end_points['Mixed_4e/Branch_2/Conv2d_0a_1x1'],64,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_3):
					branch_3=slim.max_pool2d(end_points['Mixed_4d'],[3,3],stride=1,scope='MaxPool_0a_3x3')
					branch_3=slim.conv2d(end_points['Mixed_4e/Branch_3/MaxPool_0a_3x3'],64,[1,1],stride=1,scope='Conv2d_0b_1x1')
				
				concat_point=tf.concat(axis=3,values=[branch_0,branch_1,branch_2,branch_3])
			end_points[Mixed_4e]=concat_point
			with tf.variable_scope(Mixed_4f):
				with tf.variable_scope(branch_0):
					branch_0=slim.conv2d(end_points['Mixed_4e'],256,[1,1],stride=1,scope='Conv2d_0a_1x1')
				
				with tf.variable_scope(branch_1):
					branch_1=slim.conv2d(end_points['Mixed_4e'],160,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_1=slim.conv2d(end_points['Mixed_4f/Branch_1/Conv2d_0a_1x1'],320,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_2):
					branch_2=slim.conv2d(end_points['Mixed_4e'],32,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_2=slim.conv2d(end_points['Mixed_4f/Branch_2/Conv2d_0a_1x1'],128,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_3):
					branch_3=slim.max_pool2d(end_points['Mixed_4e'],[3,3],stride=1,scope='MaxPool_0a_3x3')
					branch_3=slim.conv2d(end_points['Mixed_4f/Branch_3/MaxPool_0a_3x3'],128,[1,1],stride=1,scope='Conv2d_0b_1x1')
				
				concat_point=tf.concat(axis=3,values=[branch_0,branch_1,branch_2,branch_3])
			end_points[Mixed_4f]=concat_point
			end_points['MaxPool_5a_2x2']=slim.max_pool2d(end_points['Mixed_4f'],[3,3],stride=2,scope='MaxPool_5a_2x2')
			with tf.variable_scope(Mixed_5b):
				with tf.variable_scope(branch_0):
					branch_0=slim.conv2d(end_points['MaxPool_5a_2x2'],256,[1,1],stride=1,scope='Conv2d_0a_1x1')
				
				with tf.variable_scope(branch_1):
					branch_1=slim.conv2d(end_points['MaxPool_5a_2x2'],160,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_1=slim.conv2d(end_points['Mixed_5b/Branch_1/Conv2d_0a_1x1'],320,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_2):
					branch_2=slim.conv2d(end_points['MaxPool_5a_2x2'],32,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_2=slim.conv2d(end_points['Mixed_5b/Branch_2/Conv2d_0a_1x1'],128,[3,3],stride=1,scope='Conv2d_0a_3x3')
				
				with tf.variable_scope(branch_3):
					branch_3=slim.max_pool2d(end_points['MaxPool_5a_2x2'],[3,3],stride=1,scope='MaxPool_0a_3x3')
					branch_3=slim.conv2d(end_points['Mixed_5b/Branch_3/MaxPool_0a_3x3'],128,[1,1],stride=1,scope='Conv2d_0b_1x1')
				
				concat_point=tf.concat(axis=3,values=[branch_0,branch_1,branch_2,branch_3])
			end_points[Mixed_5b]=concat_point
			with tf.variable_scope(Mixed_5c):
				with tf.variable_scope(branch_0):
					branch_0=slim.conv2d(end_points['Mixed_5b'],384,[1,1],stride=1,scope='Conv2d_0a_1x1')
				
				with tf.variable_scope(branch_1):
					branch_1=slim.conv2d(end_points['Mixed_5b'],192,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_1=slim.conv2d(end_points['Mixed_5c/Branch_1/Conv2d_0a_1x1'],384,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_2):
					branch_2=slim.conv2d(end_points['Mixed_5b'],48,[1,1],stride=1,scope='Conv2d_0a_1x1')
					branch_2=slim.conv2d(end_points['Mixed_5c/Branch_2/Conv2d_0a_1x1'],128,[3,3],stride=1,scope='Conv2d_0b_3x3')
				
				with tf.variable_scope(branch_3):
					branch_3=slim.max_pool2d(end_points['Mixed_5b'],[3,3],stride=1,scope='MaxPool_0a_3x3')
					branch_3=slim.conv2d(end_points['Mixed_5c/Branch_3/MaxPool_0a_3x3'],128,[1,1],stride=1,scope='Conv2d_0b_1x1')
				
				concat_point=tf.concat(axis=3,values=[branch_0,branch_1,branch_2,branch_3])
			end_points[Mixed_5c]=concat_point
			end_points['AvgPool_0a_7x7']=slim.max_pool2d(end_points['Mixed_5c'],[7,7],stride=1,scope='AvgPool_0a_7x7')
			end_points['Logits/Dropout_0b']=slim.dropout(end_points['AvgPool_0a_7x7'],0.8,scope='Logits/Dropout_0b')
			with tf.variable_scope(Logits):
				with tf.variable_scope(branch_0):
					branch_0=slim.conv2d(end_points['logits'],1000,[1,1],stride=1,scope='Conv2d_0c_1x1')
					Logits=branch_0
					Logits=tf.squeeze(Logits, [1, 2], name='SpatialSqueeze')
				
				with tf.variable_scope(branch_1):
					
				with tf.variable_scope(branch_2):
					end_points['Predictions']=slim.softmax(end_points['logits'],scope='Predictions')
				
			
			end_points[Logits]=Logits
		
	
	return Logits,end_points

 
 ### change the default image_size based on the input image size specified in prototxt ### 
InceptionV1.default_image_size = 224


# The below code is applicable to any model. It is adapted from 
# https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_utils.py
def default_arg_scope(is_training=True, 
                        weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):

  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': batch_norm_updates_collections,
      # use fused batch norm if possible.
      'fused': None,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}

  # Set training state 
  with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
      # Set batch norm 
      with slim.arg_scope(
          [slim.conv2d],
          normalizer_fn=normalizer_fn,
          normalizer_params=normalizer_params):
          # Set default padding and stride
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                      stride=1, padding='SAME') as sc:
              return sc