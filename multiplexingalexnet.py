import tensorflow as tf
slim=tf.contrib.slim

def AlexNet(inputs,num_classes=1000,is_training=true,reuse=true,scope=AlexNet,config=None):
	
	
	############## template code added for multiplexing ##############
	# calculate the number of filter in a conv given config 
	selectdepth = lambda k,v: int(config[k]['ratio']*v) if config and k in config and 'ratio' in config[k] else v 
	
	# select the input tensor to a module 
	selectinput = lambda k, v: config[k]['input'] if config and k in config and 'input' in config[k] else v 
	############## end template code ##############
	
	with tf.variable_scope(scope,"Model",reuse=reuse):
		with slim.arg_scope(default_arg_scope(is_training):
			end_points= {}
			end_points['inputs']=inputs
			end_points['conv1']=slim.conv2d(end_points['inputs'],96,[11,11],stride=4,scope='conv1')
			end_points['pool1']=slim.max_pool2d(end_points['conv1'],[3,3],stride=2,scope='pool1')
			end_points['conv2']=slim.conv2d(end_points['pool1'],256,[5,5],stride=1,scope='conv2')
			end_points['pool2']=slim.max_pool2d(end_points['conv2'],[3,3],stride=2,scope='pool2')
			end_points['conv3']=slim.conv2d(end_points['pool2'],384,[3,3],stride=1,scope='conv3')
			end_points['conv4']=slim.conv2d(end_points['conv3'],384,[3,3],stride=1,scope='conv4')
			end_points['conv5']=slim.conv2d(end_points['conv4'],256,[3,3],stride=1,scope='conv5')
			end_points['pool5']=slim.max_pool2d(end_points['conv5'],[3,3],stride=2,scope='pool5')
			end_points['drop7']=slim.dropout(end_points['pool5'],0.5,scope='drop7')
			end_points['fc8']=slim.conv2d(end_points['drop7'],1000,[5,5],stride=1,scope='fc8')
			Logits=fc8
			Logits=tf.squeeze(Logits, [1, 2], name='SpatialSqueeze')
			end_points['loss']=slim.softmax(end_points['fc8'],scope='loss')
		
	
	return Logits,end_points

 
 ### change the default image_size based on the input image size specified in prototxt ### 
AlexNet.default_image_size = 224


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