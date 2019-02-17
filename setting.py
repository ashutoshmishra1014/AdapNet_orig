import tensorflow as tf
import yaml
import utils
import shutil
import os

def init():
	#training session settings
	global sess
	global config
	global class_names
	global label_values
	global num_classes
	global min_width
	global min_height
	global resize_resolution
	global data_size

	config_proto = tf.ConfigProto()
	# config_proto.gpu_options.allow_growth = True
	config_proto.gpu_options.per_process_gpu_memory_fraction=0.8
	sess=tf.Session(config=config_proto)

	#fetch config parameters
	# with open(config_filepath) as config_file:
	# 	config=yaml.load(config_file)
	with open("./config.yaml") as config_file:
		config=yaml.load(config_file)

	class_names, label_values = utils.get_label_info()
	num_classes = len(label_values)

	# resize_resolution = utils.calculate_optimal_resize_image_resolution()
	resize_resolution = [384, 768]
	data_size = config["data"]["data_size"]

	#clean up checkpoint directory
	ckpt_model_dirs = os.listdir(config["training_setting"]["checkpoints"]["save_directory"])
	for model_dir in ckpt_model_dirs:
		for ckpt_dir in os.listdir(config["training_setting"]["checkpoints"]["save_directory"]+"/"+model_dir):
			no_checkpoints = (len(os.listdir(config["training_setting"]["checkpoints"]["save_directory"]+"/"+model_dir+"/"+ckpt_dir)) == 3) or (os.listdir(config["training_setting"]["checkpoints"]["save_directory"]+"/"+model_dir+"/"+ckpt_dir) == [])
			if no_checkpoints:
				shutil.rmtree(config["training_setting"]["checkpoints"]["save_directory"]+"/"+model_dir+"/"+ckpt_dir)				




