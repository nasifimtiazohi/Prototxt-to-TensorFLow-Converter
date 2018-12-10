import tensorflow as tf
slim=tf.contrib.slim

def LeNet(inputs_1,num_classes=50,is_training=true,reuse=true,scope=LeNet):
	with tf.variable_scope(scope,"Model",reuse=reuse):
		with slim.arg_scope(default_arg_scope(is_training):
			end_points= {}
			end_points['inputs_1']=inputs_1
			end_points['conv1']=slim.conv2d(end_points['inputs_1'],20,[5,5],stride=1,scope='conv1')
			end_points['pool1']=slim.max_pool2d(end_points['conv1'],[2,2],stride=2,scope='pool1')
			end_points['conv2']=slim.conv2d(end_points['pool1'],50,[5,5],stride=1,scope='conv2')
			Logits=conv2
			Logits=tf.squeeze(Logits, [1, 2], name='SpatialSqueeze')
			end_points['prob']=slim.softmax(end_points['conv2'],scope='prob')
		
	
	return Logits,end_points

 
 ### change the default image_size based on the input image size specified in prototxt ### 
LeNet.default_image_size = 28


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