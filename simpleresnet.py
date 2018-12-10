import tensorflow as tf
slim=tf.contrib.slim

def ResNet-50(data,num_classes=1000,is_training=true,reuse=true,scope=ResNet-50):
	with tf.variable_scope(scope,"Model",reuse=reuse):
		with slim.arg_scope(default_arg_scope(is_training):
			end_points= {}
			end_points['data']=data
			end_points['conv1']=slim.conv2d(end_points['data'],64,[7,7],stride=2,scope='conv1')
			end_points['pool1']=slim.max_pool2d(end_points['conv1'],[3,3],stride=2,scope='pool1')
			with tf.variable_scope(Logits):
				with tf.variable_scope(branch_0):
					branch_0=slim.conv2d(end_points['pool1'],256,[1,1],stride=1,scope='res2a_branch1')
					with tf.variable_scope(Logits):
						with tf.variable_scope(branch_0):
							branch_0=slim.conv2d(end_points['res2a'],64,[1,1],stride=1,scope='res2b_branch2a')
							branch_0=slim.conv2d(end_points['res2b_branch2a'],64,[3,3],stride=1,scope='res2b_branch2b')
							branch_0=slim.conv2d(end_points['res2b_branch2b'],256,[1,1],stride=1,scope='res2b_branch2c')
							with tf.variable_scope(Logits):
								with tf.variable_scope(branch_0):
									branch_0=slim.conv2d(end_points['res2b'],64,[1,1],stride=1,scope='res2c_branch2a')
									branch_0=slim.conv2d(end_points['res2c_branch2a'],64,[3,3],stride=1,scope='res2c_branch2b')
									branch_0=slim.conv2d(end_points['res2c_branch2b'],256,[1,1],stride=1,scope='res2c_branch2c')
									with tf.variable_scope(Logits):
										with tf.variable_scope(branch_0):
											branch_0=slim.conv2d(end_points['res2c'],512,[1,1],stride=2,scope='res3a_branch1')
											with tf.variable_scope(Logits):
												with tf.variable_scope(branch_0):
													branch_0=slim.conv2d(end_points['res3a'],128,[1,1],stride=1,scope='res3b_branch2a')
													branch_0=slim.conv2d(end_points['res3b_branch2a'],128,[3,3],stride=1,scope='res3b_branch2b')
													branch_0=slim.conv2d(end_points['res3b_branch2b'],512,[1,1],stride=1,scope='res3b_branch2c')
													with tf.variable_scope(Logits):
														with tf.variable_scope(branch_0):
															branch_0=slim.conv2d(end_points['res3b'],128,[1,1],stride=1,scope='res3c_branch2a')
															branch_0=slim.conv2d(end_points['res3c_branch2a'],128,[3,3],stride=1,scope='res3c_branch2b')
															branch_0=slim.conv2d(end_points['res3c_branch2b'],512,[1,1],stride=1,scope='res3c_branch2c')
															with tf.variable_scope(Logits):
																with tf.variable_scope(branch_0):
																	branch_0=slim.conv2d(end_points['res3c'],128,[1,1],stride=1,scope='res3d_branch2a')
																	branch_0=slim.conv2d(end_points['res3d_branch2a'],128,[3,3],stride=1,scope='res3d_branch2b')
																	branch_0=slim.conv2d(end_points['res3d_branch2b'],512,[1,1],stride=1,scope='res3d_branch2c')
																	with tf.variable_scope(Logits):
																		with tf.variable_scope(branch_0):
																			branch_0=slim.conv2d(end_points['res3d'],1024,[1,1],stride=2,scope='res4a_branch1')
																			with tf.variable_scope(Logits):
																				with tf.variable_scope(branch_0):
																					branch_0=slim.conv2d(end_points['res4a'],256,[1,1],stride=1,scope='res4b_branch2a')
																					branch_0=slim.conv2d(end_points['res4b_branch2a'],256,[3,3],stride=1,scope='res4b_branch2b')
																					branch_0=slim.conv2d(end_points['res4b_branch2b'],1024,[1,1],stride=1,scope='res4b_branch2c')
																					with tf.variable_scope(Logits):
																						with tf.variable_scope(branch_0):
																							branch_0=slim.conv2d(end_points['res4b'],256,[1,1],stride=1,scope='res4c_branch2a')
																							branch_0=slim.conv2d(end_points['res4c_branch2a'],256,[3,3],stride=1,scope='res4c_branch2b')
																							branch_0=slim.conv2d(end_points['res4c_branch2b'],1024,[1,1],stride=1,scope='res4c_branch2c')
																							with tf.variable_scope(Logits):
																								with tf.variable_scope(branch_0):
																									branch_0=slim.conv2d(end_points['res4c'],256,[1,1],stride=1,scope='res4d_branch2a')
																									branch_0=slim.conv2d(end_points['res4d_branch2a'],256,[3,3],stride=1,scope='res4d_branch2b')
																									branch_0=slim.conv2d(end_points['res4d_branch2b'],1024,[1,1],stride=1,scope='res4d_branch2c')
																									with tf.variable_scope(Logits):
																										with tf.variable_scope(branch_0):
																											branch_0=slim.conv2d(end_points['res4d'],256,[1,1],stride=1,scope='res4e_branch2a')
																											branch_0=slim.conv2d(end_points['res4e_branch2a'],256,[3,3],stride=1,scope='res4e_branch2b')
																											branch_0=slim.conv2d(end_points['res4e_branch2b'],1024,[1,1],stride=1,scope='res4e_branch2c')
																											with tf.variable_scope(Logits):
																												with tf.variable_scope(branch_0):
																													branch_0=slim.conv2d(end_points['res4e'],256,[1,1],stride=1,scope='res4f_branch2a')
																													branch_0=slim.conv2d(end_points['res4f_branch2a'],256,[3,3],stride=1,scope='res4f_branch2b')
																													branch_0=slim.conv2d(end_points['res4f_branch2b'],1024,[1,1],stride=1,scope='res4f_branch2c')
																													with tf.variable_scope(Logits):
																														with tf.variable_scope(branch_0):
																															branch_0=slim.conv2d(end_points['res4f'],2048,[1,1],stride=2,scope='res5a_branch1')
																															with tf.variable_scope(Logits):
																																with tf.variable_scope(branch_0):
																																	branch_0=slim.conv2d(end_points['res5a'],512,[1,1],stride=1,scope='res5b_branch2a')
																																	branch_0=slim.conv2d(end_points['res5b_branch2a'],512,[3,3],stride=1,scope='res5b_branch2b')
																																	branch_0=slim.conv2d(end_points['res5b_branch2b'],2048,[1,1],stride=1,scope='res5b_branch2c')
																																	with tf.variable_scope(Logits):
																																		with tf.variable_scope(branch_0):
																																			branch_0=slim.conv2d(end_points['res5b'],512,[1,1],stride=1,scope='res5c_branch2a')
																																			branch_0=slim.conv2d(end_points['res5c_branch2a'],512,[3,3],stride=1,scope='res5c_branch2b')
																																			branch_0=slim.conv2d(end_points['res5c_branch2b'],2048,[1,1],stride=1,scope='res5c_branch2c')
																																			branch_0=slim.max_pool2d(end_points['res5c'],[7,7],stride=1,scope='pool5')
																																			Logits=branch_0
																																			Logits=tf.squeeze(Logits, [1, 2], name='SpatialSqueeze')
																																			end_points['prob']=slim.softmax(end_points['fc1000'],scope='prob')
																																		
																																		with tf.variable_scope(branch_1):
																																			
																																	
																																	end_points[Logits]=Logits
																																	branch_0=concat_point
																																with tf.variable_scope(branch_1):
																																	
																															
																															end_points[Logits]=Logits
																															branch_0=concat_point
																														with tf.variable_scope(branch_1):
																															branch_1=slim.conv2d(end_points['res4f'],512,[1,1],stride=2,scope='res5a_branch2a')
																															branch_1=slim.conv2d(end_points['res5a_branch2a'],512,[3,3],stride=1,scope='res5a_branch2b')
																															branch_1=slim.conv2d(end_points['res5a_branch2b'],2048,[1,1],stride=1,scope='res5a_branch2c')
																														
																													
																													end_points[Logits]=Logits
																													branch_0=concat_point
																												with tf.variable_scope(branch_1):
																													
																											
																											end_points[Logits]=Logits
																											branch_0=concat_point
																										with tf.variable_scope(branch_1):
																											
																									
																									end_points[Logits]=Logits
																									branch_0=concat_point
																								with tf.variable_scope(branch_1):
																									
																							
																							end_points[Logits]=Logits
																							branch_0=concat_point
																						with tf.variable_scope(branch_1):
																							
																					
																					end_points[Logits]=Logits
																					branch_0=concat_point
																				with tf.variable_scope(branch_1):
																					
																			
																			end_points[Logits]=Logits
																			branch_0=concat_point
																		with tf.variable_scope(branch_1):
																			branch_1=slim.conv2d(end_points['res3d'],256,[1,1],stride=2,scope='res4a_branch2a')
																			branch_1=slim.conv2d(end_points['res4a_branch2a'],256,[3,3],stride=1,scope='res4a_branch2b')
																			branch_1=slim.conv2d(end_points['res4a_branch2b'],1024,[1,1],stride=1,scope='res4a_branch2c')
																		
																	
																	end_points[Logits]=Logits
																	branch_0=concat_point
																with tf.variable_scope(branch_1):
																	
															
															end_points[Logits]=Logits
															branch_0=concat_point
														with tf.variable_scope(branch_1):
															
													
													end_points[Logits]=Logits
													branch_0=concat_point
												with tf.variable_scope(branch_1):
													
											
											end_points[Logits]=Logits
											branch_0=concat_point
										with tf.variable_scope(branch_1):
											branch_1=slim.conv2d(end_points['res2c'],128,[1,1],stride=2,scope='res3a_branch2a')
											branch_1=slim.conv2d(end_points['res3a_branch2a'],128,[3,3],stride=1,scope='res3a_branch2b')
											branch_1=slim.conv2d(end_points['res3a_branch2b'],512,[1,1],stride=1,scope='res3a_branch2c')
										
									
									end_points[Logits]=Logits
									branch_0=concat_point
								with tf.variable_scope(branch_1):
									
							
							end_points[Logits]=Logits
							branch_0=concat_point
						with tf.variable_scope(branch_1):
							
					
					end_points[Logits]=Logits
					branch_0=concat_point
				with tf.variable_scope(branch_1):
					branch_1=slim.conv2d(end_points['pool1'],64,[1,1],stride=1,scope='res2a_branch2a')
					branch_1=slim.conv2d(end_points['res2a_branch2a'],64,[3,3],stride=1,scope='res2a_branch2b')
					branch_1=slim.conv2d(end_points['res2a_branch2b'],256,[1,1],stride=1,scope='res2a_branch2c')
				
			
			end_points[Logits]=Logits
		
	
	return Logits,end_points

 
 ### change the default image_size based on the input image size specified in prototxt ### 
ResNet-50.default_image_size = 0


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