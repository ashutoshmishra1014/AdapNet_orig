import tensorflow as tf
import numpy as np
import os
import datetime
from NetworkBuilder import NetworkBuilder
import setting


class Adapnet():
    def __init__(self):
    	pass

    def _setup(self, input, is_training):

    	num_classes = setting.num_classes
    	nb = NetworkBuilder(is_training)
    	# keep_prob = tf.cond(is_training, lambda: 0.3, lambda: 1.0)

    	model = input

    	model = nb.attach_conv_layer(model, output_size=32, feature_size=[3,3], strides=[1,1,1,1], padding=1, use_bias=True)
    	model = nb.attach_batch_norm_layer(model)
    	model = nb.attach_relu_layer(model)
    	self.conv_3x3_out = model

    	model = nb.attach_conv_layer(model, output_size=64, feature_size=[7,7], strides=[1,2,2,1], padding=3, use_bias=True)
    	model = nb.attach_batch_norm_layer(model)
    	model = nb.attach_relu_layer(model)
    	self.conv_7x7_out = model

    	model = nb.attach_pooling_layer(model, ksize=[1,2,2,1], strides=[1,2,2,1], padding=0)
    	self.max_pool_out = model

    	model = nb.attach_resnet_block_2(model, d1=64, d2=256, strides=[1,1,1,1])
    	model = nb.attach_resnet_block_1(model, d1=64, d2=256)
    	model = nb.attach_resnet_block_1(model, d1=64, d2=256)
    	self.m_b1_out = model

    	model = nb.attach_resnet_block_2(model, d1=128, d2=512, strides=[1,2,2,1])
    	model = nb.attach_resnet_block_1(model, d1=128, d2=512)
    	model = nb.attach_resnet_block_1(model, d1=128, d2=512)
    	model = nb.attach_multiscale_block_1(model, d1=128, d2=512, d3=128, p=1, d=2)
    	self.m_b2_out = model

    	#skip branch
    	skip_connection = model
    	# self.skip = model

    	model = nb.attach_resnet_block_2(model, d1=256, d2=1024, strides=[1,2,2,1])
    	model = nb.attach_resnet_block_1(model, d1=256, d2=1024)
    	model = nb.attach_multiscale_block_1(model, d1=256, d2=1024, d3=256, p=1, d=2)
    	model = nb.attach_multiscale_block_1(model, d1=256, d2=1024, d3=256, p=1, d=4)
    	model = nb.attach_multiscale_block_1(model, d1=256, d2=1024, d3=256, p=1, d=8)
    	model = nb.attach_multiscale_block_1(model, d1=256, d2=1024, d3=256, p=1, d=16)
    	self.m_b3_out = model

    	model = nb.attach_multiscale_block_2(model, d1=512, d2=2048, d3=512, p=2, d=4)
    	model = nb.attach_multiscale_block_1(model, d1=512, d2=2048, d3=512, p=2, d=8)
    	model = nb.attach_multiscale_block_1(model, d1=512, d2=2048, d3=512, p=2, d=16)
    	# model = nb.add_dropout(model, keep_prob)
    	self.m_b4_out = model

    	model = nb.attach_conv_layer(model, output_size=num_classes, feature_size=[1,1], strides=[1,1,1,1], padding=0, use_bias=True)
    	model = nb.attach_batch_norm_layer(model)
    	self.out = model

    	model = nb.attach_conv_transpose_layer(model, output_size=num_classes*2, feature_size=[4,4], strides=[2,2], padding=1, use_bias=True, scale=2)
    	# model = nb.attach_batch_norm_layer(model)
    	#adding missing batch norm layer
    	self.deconv_up1 = model

    	#is saale ko check kr, kahi naatak na kr de baad mein
    	skip_connection = nb.attach_conv_layer(skip_connection, output_size=num_classes*2, feature_size=[1,1], strides=[1,1,1,1],\
										padding=0, use_bias=True)
    	skip_connection = nb.attach_batch_norm_layer(skip_connection)
    	self.skip = skip_connection

    	model = nb.attach_element_wise_sum(model, skip_connection)
    	self.up1 = model

    	model = nb.attach_conv_transpose_layer(model, output_size=num_classes, feature_size=[16,16], strides=[8,8], padding=4, use_bias=True, scale=8)
    	model = nb.attach_batch_norm_layer(model)
    	self.deconv_up2 = model

    	#could be removed
    	model = nb.attach_cropping_layer(input, model)
    	self.cropped_deconv_up2 = model

    	#could be removed
    	model = nb.attach_conv_layer(model, output_size=num_classes, feature_size=[1,1], strides=[1,1,1,1], padding="SAME",\
									use_bias=False)

    	self.up2 = model
    	return model


def get_model(name=None):
	if name == "Adapnet":
		return Adapnet()