import tensorflow as tf
import setting
import numpy as np

padding_scheme="caffe"


class NetworkBuilder:
	def __init__(self, is_training):
		self.training = is_training
		self.conv_weights_name_index = 0
		self.conv_bias_weights_name_index = 0
		self.atrous_conv_weights_name_index = 0
		self.atrous_conv_bias_weights_name_index = 0
		self.deconv_layer_name_index = 0
		self.deconv_weights = 0
		self.deconv_biases = 0
		self.batch_norm_layer_name_index = 0
		self.dense_weights_name_index = 0
		self.dense_bias_name_index = 0
		self.element_wise_sum_name_index = 0
		self.concat_layer_name_index = 0

	def attach_conv_layer(self, input_layer, output_size=32, feature_size=[5,5],\
						strides=[1,1,1,1], padding=None, dilations=[1, 1, 1, 1], use_bias=True, data_format='NHWC',summary=False):
		
		with tf.name_scope("Convolution") as scope:
			input_size = input_layer.get_shape().as_list()[-1]
			with tf.variable_scope("Conv_variable"):
				weights = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=([feature_size[0],\
							feature_size[1], input_size, output_size]), name="conv_weights_"+str(self.conv_weights_name_index))
				if summary:
					tf.summary.histogram(weights.name, weights)

				if padding_scheme=="caffe" and (type(padding)==type(0)):
					input_layer = tf.pad(input_layer, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
					conv = tf.nn.conv2d(input_layer, weights, strides=strides, dilations=dilations, padding="VALID", data_format=data_format)
				else:
					conv = tf.nn.conv2d(input_layer, weights, strides=strides, dilations=dilations, padding=padding, data_format=data_format)

				if use_bias:
					biases = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[output_size],\
											name="conv_biases_"+str(self.conv_bias_weights_name_index))
					conv = conv+biases

				self.conv_weights_name_index+=1
				self.conv_bias_weights_name_index+=1

		return conv


	def attach_atrous_conv_layer(self, input_layer, output_size=32, feature_size=[5,5],\
						padding=None, rate=1, use_bias=True, summary=False):

		with tf.name_scope("AtrousConvolution") as scope:
			input_size = input_layer.get_shape().as_list()[-1]
			with tf.variable_scope("atrous_conv_variable") as scope:
				weights = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=([feature_size[0],\
							feature_size[1],input_size,output_size]), name="atrous_conv_weights_"+str(self.atrous_conv_weights_name_index))
				if summary:
					tf.summary.histogram(weights.name, weights)

				if padding_scheme=="caffe" and (type(padding)==type(0)):
					input_layer = tf.pad(input_layer, [[0,0],[padding,padding],[padding,padding],[0,0]], "CONSTANT")
					atrous_conv = tf.nn.atrous_conv2d(input_layer, weights, rate=rate, padding="VALID")
				else:
					atrous_conv = tf.nn.atrous_conv2d(input_layer, weights, rate=rate, padding=padding)

				if use_bias:
					biases = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[output_size],\
												name="atrous_conv_biases_"+str(self.atrous_conv_bias_weights_name_index))
					atrous_conv = atrous_conv+biases

				self.atrous_conv_weights_name_index+=1
				self.atrous_conv_bias_weights_name_index+=1

		return atrous_conv



#this code needs to be changed to use tf.nn.con2d_transpose
	def attach_conv_transpose_layer(self, input_layer, output_size=26, feature_size=[4,4],\
						strides=[1,1], padding=None, use_bias=True, scale=None, data_format='channels_last', summary=False):
		
		with tf.name_scope("Deconvolution") as scope:
			input_size = input_layer.get_shape().as_list()

			with tf.variable_scope("Deconv_variable"):
				
				# weights = tf.get_variable(initializer=tf.initializers.bilinear(scale=scale), shape=([feature_size[0],feature_size[1], output_size, input_size[-1]]), name="deconv_weights_"+str(self.deconv_weights))
				
				# if summary:
				# 	tf.summary.histogram(weights.name, weights)

				# output_shape = tf.stack([tf.shape(input_layer)[0], input_size[1]*scale, input_size[2]*scale, output_size])
				
				# strides.insert(0,1)
				# strides.append(1)

				# if padding_scheme=="caffe":
				# 	input_layer = tf.pad(input_layer, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
				# 	conv_transpose = tf.nn.conv2d_transpose(input_layer, weights, output_shape=output_shape, strides=strides, padding="VALID")
				# else:
				# 	conv_transpose = tf.nn.conv2d_transpose(input_layer, weights, output_shape=output_shape, strides=strides, padding=padding)

				# if use_bias:
				# 	biases = tf.get_variable(initializer=tf.zeros_initializer(), shape=[output_size],\
				# 							name="deconv_biases_"+str(self.deconv_biases))
				# 	conv_transpose = conv_transpose+biases

				# self.deconv_weights+=1
				# self.deconv_biases+=1

				if padding_scheme=="caffe":
					# input_layer = tf.pad(input_layer, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
					if use_bias:
						conv_transpose = tf.layers.conv2d_transpose(inputs=input_layer, filters=output_size, kernel_size=feature_size, \
										strides=strides, padding='SAME', use_bias=use_bias, kernel_initializer=tf.initializers.bilinear(scale=scale),\
										bias_initializer=tf.zeros_initializer(), data_format=data_format, name="Deconv_layer_"+str(self.deconv_layer_name_index))
					else:
						conv_transpose = tf.layers.conv2d_transpose(inputs=input_layer, filters=output_size, kernel_size=feature_size, \
										strides=strides, padding='SAME', use_bias=use_bias, kernel_initializer=tf.initializers.bilinear(scale=scale),\
										data_format=data_format, name="Deconv_layer"+str(self.deconv_layer_name_index))
				else:
					if use_bias:
						conv_transpose = tf.layers.conv2d_transpose(inputs=input_layer, filters=output_size, kernel_size=feature_size, \
											strides=strides, padding=padding, use_bias=use_bias, kernel_initializer=tf.initializers.bilinear(scale=scale),\
											bias_initializer=tf.zeros_initializer(), data_format=data_format, name="Deconv_layer_"+str(self.deconv_layer_name_index))
					else:
						conv_transpose = tf.layers.conv2d_transpose(inputs=input_layer, filters=output_size, kernel_size=feature_size, \
										strides=strides, padding=padding, use_bias=use_bias, kernel_initializer=tf.initializers.bilinear(scale=scale),\
										data_format=data_format, name="Deconv_layer_"+str(self.deconv_layer_name_index))
					#summary code to be written

				self.deconv_layer_name_index+=1

		return conv_transpose


	def Upsampling(inputs,scale):
	    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])


	def attach_pooling_layer(self, input_layer, ksize=[1,1,1,1], strides=[1,1,1,1], padding=None, data_format='NHWC'):
		#for caffe padding scheme, padding changed to SAME from VALID as it was producing wrong result
		with tf.name_scope("Pooling") as scope:
				if padding_scheme=="caffe":
					input_layer = tf.pad(input_layer, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
					return tf.nn.max_pool(input_layer, ksize=ksize, strides=strides, padding="SAME", data_format=data_format)
				else:
					return tf.nn.max_pool(input_layer, ksize=ksize, strides=strides, padding="SAME", data_format=data_format)


	def attach_batch_norm_layer(self, input_layer):
		with tf.name_scope("Batch_Norm") as scope:
			batch_norm = tf.layers.batch_normalization(input_layer, training=self.training, name="batch_norm_"+str(self.batch_norm_layer_name_index),\
													fused=True)
			self.batch_norm_layer_name_index+=1

			# print_training_status = tf.Print(self.training, [self.training], message="printing batch norm layer training status")
			return batch_norm


		self.batch_norm_layer_name_index+=1

	def attach_relu_layer(self, input_layer):

		with tf.name_scope("Activation") as scope:
			return tf.nn.relu(input_layer)


	def attach_sigmoid_layer(self, input_layer):

		with tf.name_scope("Activation") as scope:
			return tf.nn.sigmoid(input_layer)


	def attach_softmax_layer(self, input_layer):

		with tf.name_scope("Activation") as scope:
			return tf.nn.softmax(input_layer)


	def attach_leaky_relu(self, input_layer, alpha=0.2):

		with tf.name_scope("Activation") as scope:
			return tf.nn.leaky_relu(input_layer, alpha=alpha)


	def flatten(self, input_layer):

		with tf.name_scope("Flatten") as scope:
			# input_size = input_layer.get_shape().as_list()
			# new_size = input_size[-1]*input_size[-2]*input_size[-3]
			# return tf.reshape(input_layer, [-1,new_size])
			# dim = np.prod(input_layer.get_shape().as_list()[1:])
			# dim = tf.reduce_prod(tf.shape(input_layer)[1:])
			# return tf.reshape(input_layer, [-1, dim])

			return tf.layers.flatten(input_layer)


	def attach_dense_layer(self, input_layer, size, summary=False):

		with tf.name_scope("Dense") as scope:
			input_size = input_layer.get_shape().as_list()[-1]
			with tf.variable_scope("Dense"):
				weights = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),\
					shape=[input_size, size], name="dense_weights_"+str(self.dense_weights_name_index))
				biases = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),\
					shape=[size], name="dense_biases_"+str(self.dense_bias_name_index))
				dense = tf.matmul(input_layer, weights) + biases

		if summary:
			tf.summary.histogram(weights.name, weights)


		self.dense_weights_name_index+=1
		self.dense_bias_name_index+=1

		return dense


	def attach_element_wise_sum(self, branch1, branch2):

		assert_op = tf.assert_equal(tf.shape(branch1), tf.shape(branch2))

		with tf.control_dependencies([assert_op]):
			branch1 = tf.slice(branch1, [0,0,0,0], tf.shape(branch1))
			branch2 = tf.slice(branch2, [0,0,0,0], tf.shape(branch1))

		with tf.name_scope("Elementwise_Sum") as scope:
			elemwise_sum = tf.add(branch1, branch2, name="elementwise_sum_"+str(self.element_wise_sum_name_index))

		self.element_wise_sum_name_index+=1
		return elemwise_sum


	def attach_concat_layer(self, values, axis=-1):

		with tf.name_scope("Concat") as scope:
			concat = tf.concat(values, axis=axis, name="concat_"+str(self.concat_layer_name_index))
		
		self.concat_layer_name_index+=1
		return concat


	def add_dropout(self, input_layer, keep_prob):

		return tf.nn.dropout(input_layer, keep_prob)


	def spatial_dropout(self, input_layer, keep_prob):

		num_feature_maps = [tf.shape(input_layer)[0], tf.shape(input_layer)[3]]
		random_tensor = keep_prob
		random_tensor += tf.random_uniform(num_feature_maps, dtype=input_layer.dtype)
		binary_tensor = tf.floor(random_tensor)
		binary_tensor = tf.reshape(binary_tensor, [-1, 1, 1, tf.shape(x)[3]])
		output = tf.div(input_layer, keep_prob) * binary_tensor

		return output


	def reshape(self, input_layer, out_shape):

		pass

	def rename_tensor(self, input_layer, name):

		return tf.identity(input_layer, name=name)

	def attach_resnet_block_1(self, input_layer, d1=64, d2=256, strides=[1,1,1,1]):

		with tf.name_scope("Resnet_block_1") as scope:
			block_1 = self.attach_conv_layer(input_layer, output_size=d1, feature_size=[1,1], strides=[1,1,1,1], padding=0, use_bias=False, summary=True)
			block_1 = self.attach_batch_norm_layer(block_1)
			block_1 = self.attach_relu_layer(block_1)

			block_1 = self.attach_conv_layer(block_1, output_size=d1, feature_size=[3,3], strides=[1,1,1,1], padding=1, use_bias=False, summary=True)
			block_1 = self.attach_batch_norm_layer(block_1)
			block_1 = self.attach_relu_layer(block_1)

			block_1 = self.attach_conv_layer(block_1, output_size=d2, feature_size=[1,1], strides=[1,1,1,1], padding=0, use_bias=False, summary=True)
			block_1 = self.attach_batch_norm_layer(block_1)
			
			block_1 = self.attach_relu_layer(input_layer+block_1)

		return block_1

	def attach_resnet_block_2(self, input_layer, d1=64, d2=256, strides=[1,1,1,1]):

		with tf.name_scope("Resnet_block_2") as scope:
			#parallel branch1
			branch_1_block_2 = self.attach_conv_layer(input_layer, output_size=d2, feature_size=[1,1], strides=strides, padding=0, use_bias=False, summary=True)
			branch_1_block_2 = self.attach_batch_norm_layer(branch_1_block_2)
			

			#parallel branch2
			branch_2_block_2 = self.attach_conv_layer(input_layer, output_size=d1, feature_size=[1,1], strides=strides, padding=0, use_bias=False, summary=True)
			branch_2_block_2 = self.attach_batch_norm_layer(branch_2_block_2)
			branch_2_block_2 = self.attach_relu_layer(branch_2_block_2)

			branch_2_block_2 = self.attach_conv_layer(branch_2_block_2, output_size=d1, feature_size=[3,3], strides=[1,1,1,1], padding=1, use_bias=False, summary=True)
			branch_2_block_2 = self.attach_batch_norm_layer(branch_2_block_2)
			branch_2_block_2 = self.attach_relu_layer(branch_2_block_2)

			branch_2_block_2 = self.attach_conv_layer(branch_2_block_2, output_size=d2, feature_size=[1,1], strides=[1,1,1,1], padding=0, use_bias=False, summary=True)
			branch_2_block_2 = self.attach_batch_norm_layer(branch_2_block_2)
			

			block_2 = self.attach_relu_layer(branch_1_block_2+branch_2_block_2)

		return block_2

	def attach_concatenation_layer(self, tensor1, tensor2, axis=None):
		return tf.concat([tensor1, tensor2], axis=axis)

	def attach_cropping_layer(self, input_data_tensor, pred_tensor):
		#the assumption is that the input data will always be smaller in size than the preds in cases were both don't match in size
		
		input_data_shape = input_data_tensor.get_shape().as_list()
		pred_shape = pred_tensor.get_shape().as_list()
		
		# height = (pred_shape[1]-input_data_shape[1])/2
		# width = (pred_shape[2]-input_data_shape[2])/2

		center_height = tf.cast((tf.shape(pred_tensor)[1]-tf.shape(input_data_tensor)[1])/2, tf.int32)
		center_width = tf.cast((tf.shape(pred_tensor)[2]-tf.shape(input_data_tensor)[2])/2, tf.int32)

		cropped_pred = tf.slice(pred_tensor, [0,center_height,center_width,0], tf.shape(input_data_tensor))

		# return center_height, center_width
		return cropped_pred

	def rename_tensor(self, tensor, name):
		return tf.identity(tensor, name=name)

	def attach_multiscale_block_1(self, input_layer, d1=256, d2=1024, d3=256, p=1, d=2, strides=[1,1,1,1]):

		with tf.name_scope("Multiscale_block_1") as scope:
			#parallel branch1
			ms_branch_1_block_1 = self.attach_conv_layer(input_layer, output_size=d1, feature_size=[1,1], strides=[1,1,1,1], padding=0, use_bias=False, summary=True)
			ms_branch_1_block_1 = self.attach_batch_norm_layer(ms_branch_1_block_1)
			ms_branch_1_block_1 = self.attach_relu_layer(ms_branch_1_block_1)

			#parallel branch1_1
			# ms_branch_1_1_block_1 = self.attach_conv_layer(ms_branch_1_block_1, output_size=int(d3/2), feature_size=[3,3], strides=[1,1,1,1], padding=p, dilations=[1,p,p,1], use_bias=False, summary=True)
			ms_branch_1_1_block_1 = self.attach_atrous_conv_layer(ms_branch_1_block_1, output_size=int(d3/2), feature_size=[3,3], padding=p, rate=p, use_bias=False, summary=True)
			ms_branch_1_1_block_1 = self.attach_batch_norm_layer(ms_branch_1_1_block_1)
			ms_branch_1_1_block_1 = self.attach_relu_layer(ms_branch_1_1_block_1)

			#parallel branch1_2
			# ms_branch_1_2_block_1 = self.attach_conv_layer(ms_branch_1_block_1, output_size=int(d3/2), feature_size=[3,3], strides=[1,1,1,1], padding=d, dilations=[1,d,d,1], use_bias=True, summary=True)
			ms_branch_1_2_block_1 = self.attach_atrous_conv_layer(ms_branch_1_block_1, output_size=int(d3/2), feature_size=[3,3], padding=d, rate=d, use_bias=True, summary=True)
			ms_branch_1_2_block_1 = self.attach_batch_norm_layer(ms_branch_1_2_block_1)
			ms_branch_1_2_block_1 = self.attach_relu_layer(ms_branch_1_2_block_1)
			
			
			assert_op = tf.assert_equal(tf.shape(ms_branch_1_1_block_1), tf.shape(ms_branch_1_2_block_1))

			with tf.control_dependencies([assert_op]):
				ms_branch_1_1_block_1 = tf.slice(ms_branch_1_1_block_1, [0,0,0,0], tf.shape(ms_branch_1_1_block_1))
				ms_branch_1_2_block_1 = tf.slice(ms_branch_1_2_block_1, [0,0,0,0], tf.shape(ms_branch_1_1_block_1))
				ms_branch_1_block_1 = tf.concat([ms_branch_1_1_block_1, ms_branch_1_2_block_1], axis=-1)

			ms_branch_1_block_1 = self.attach_conv_layer(ms_branch_1_block_1, output_size=d2, feature_size=[1,1], strides=[1,1,1,1], padding=0, use_bias=False, summary=True)
			ms_branch_1_block_1 = self.attach_batch_norm_layer(ms_branch_1_block_1)

			#parallel branch2 is a skip connection for input_layer
			assert_op_1 = tf.assert_equal(tf.shape(ms_branch_1_block_1), tf.shape(input_layer))

			with tf.control_dependencies([assert_op_1]):
				ms_branch_1_block_1 = tf.slice(ms_branch_1_block_1, [0,0,0,0], tf.shape(ms_branch_1_block_1))
				input_layer = tf.slice(input_layer, [0,0,0,0], tf.shape(ms_branch_1_block_1))
			
			ms_block_1 = self.attach_relu_layer(ms_branch_1_block_1+input_layer)

		return ms_block_1


	def attach_multiscale_block_2(self, input_layer, d1=512, d2=2048, d3=512, p=2, d=4, strides=[1,1,1,1]):

		with tf.name_scope("Multiscale_block_2") as scope:
			#parallel branch1
			ms_branch_1_block_2 = self.attach_conv_layer(input_layer, output_size=d1, feature_size=[1,1], strides=[1,1,1,1], padding=0, use_bias=False, summary=True)
			ms_branch_1_block_2 = self.attach_batch_norm_layer(ms_branch_1_block_2)
			ms_branch_1_block_2 = self.attach_relu_layer(ms_branch_1_block_2)

			#parallel branch1_1
			# ms_branch_1_1_block_2 = self.attach_conv_layer(ms_branch_1_block_2, output_size=int(d3/2), feature_size=[3,3], strides=[1,1,1,1], padding=p, dilations=[1,p,p,1], use_bias=False, summary=True)
			ms_branch_1_1_block_2 = self.attach_atrous_conv_layer(ms_branch_1_block_2, output_size=int(d3/2), feature_size=[3,3], padding=p, rate=p, use_bias=False, summary=True)
			ms_branch_1_1_block_2 = self.attach_batch_norm_layer(ms_branch_1_1_block_2)
			ms_branch_1_1_block_2 = self.attach_relu_layer(ms_branch_1_1_block_2)

			#parallel branch1_2
			# ms_branch_1_2_block_2 = self.attach_conv_layer(ms_branch_1_block_2, output_size=int(d3/2), feature_size=[3,3], strides=[1,1,1,1], padding=d, dilations=[1,d,d,1], use_bias=True, summary=True)
			ms_branch_1_2_block_2 = self.attach_atrous_conv_layer(ms_branch_1_block_2, output_size=int(d3/2), feature_size=[3,3], padding=d, rate=d, use_bias=True, summary=True)
			ms_branch_1_2_block_2 = self.attach_batch_norm_layer(ms_branch_1_2_block_2)
			ms_branch_1_2_block_2 = self.attach_relu_layer(ms_branch_1_2_block_2)

			assert_op = tf.assert_equal(tf.shape(ms_branch_1_1_block_2), tf.shape(ms_branch_1_2_block_2))

			with tf.control_dependencies([assert_op]):
				ms_branch_1_1_block_2 = tf.slice(ms_branch_1_1_block_2, [0,0,0,0], tf.shape(ms_branch_1_1_block_2))
				ms_branch_1_2_block_2 = tf.slice(ms_branch_1_2_block_2, [0,0,0,0], tf.shape(ms_branch_1_1_block_2))
				
			ms_branch_1_block_2 = tf.concat([ms_branch_1_1_block_2, ms_branch_1_2_block_2], axis=-1)

			# ms_branch_1_block_2 = tf.concat([ms_branch_1_1_block_2,ms_branch_1_2_block_2], axis=-1)

			ms_branch_1_block_2 = self.attach_conv_layer(ms_branch_1_block_2, output_size=d2, feature_size=[1,1], strides=[1,1,1,1], padding=0, use_bias=False, summary=True)
			ms_branch_1_block_2 = self.attach_batch_norm_layer(ms_branch_1_block_2)

			# #parallel branch2
			ms_branch_2_block_2 = self.attach_conv_layer(input_layer, output_size=d2, feature_size=[1,1], strides=[1,1,1,1], padding=0, use_bias=False, summary=True)
			ms_branch_2_block_2 = self.attach_batch_norm_layer(ms_branch_2_block_2)

			assert_op_1 = tf.assert_equal(tf.shape(ms_branch_1_block_2), tf.shape(ms_branch_2_block_2))

			with tf.control_dependencies([assert_op_1]):
				ms_branch_1_block_2 = tf.slice(ms_branch_1_block_2, [0,0,0,0], tf.shape(ms_branch_1_block_2))
				ms_branch_2_block_2 = tf.slice(ms_branch_2_block_2, [0,0,0,0], tf.shape(ms_branch_1_block_2))
				
			ms_block_2 = self.attach_relu_layer(ms_branch_1_block_2+ms_branch_2_block_2)
			# ms_block_2 = self.attach_relu_layer(ms_block_2)

		return ms_block_2