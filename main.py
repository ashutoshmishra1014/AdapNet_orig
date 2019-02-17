import setting
from DatasetGenerator import DatasetGenerator
from DataAugmentor import DataAugmentor
from TFRecordGenerator import TFRecordGenerator
import numpy as np
import os
import tensorflow as tf
from collections import namedtuple
import model_hub
import utils
import helpers
import logging
import time
import cv2
import tf_utils
import scipy
from time import strftime, localtime
import shutil
from matplotlib.ticker import ScalarFormatter
import matplotlib
import matplotlib.pyplot as plt
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_count = 0
val_count = 0
test_count = 0
class_weights = []

def get_data_info():

	global class_weights
	config = setting.config

	if config["training_setting"]["class_balancing"]=="default":
		class_weights = utils.compute_default_class_weights(labels_dir=config["data"]["directory"] + "/train_labels")
	elif config["training_setting"]["class_balancing"]=="median":
		class_weights = utils.compute_median_class_weights(labels_dir=config["data"]["directory"] + "/train_labels")
	elif config["training_setting"]["class_balancing"]=="custom":
		class_weights = utils.compute_custom_class_weights(labels_dir=config["data"]["directory"] + "/train_labels")

	datagen = DatasetGenerator(config["data"]["directory"])

	global train_count
	global val_count
	global test_count
	train_count = len(datagen.train_files_list)
	val_count = len(datagen.val_files_list)

def check_argument_sanity():

	config = setting.config
	if (config["data_processing"]["horizontal_flip"] or config["data_processing"]["vertical_flip"] or config["data_processing"]["brightness"] or config["data_processing"]["rotation"]) \
		and (config["data_processing"]["all_augmentation"] or config["data_processing"]["random_augmentation"]):
		raise ValueError("All/random augmentation selected along with one or more individual augmentations, select only one type")
	elif config["data_processing"]["all_augmentation"] and config["data_processing"]["random_augmentation"]:
		raise ValueError("All augmentation can not be selected along with random augmentation")
	else:
		return True

def dataset_Filename():
	return_dataset=()

	if train_count:
		train_generator = datagen.build_train_generator
		train_dataset = tf.data.Dataset.from_generator(train_generator, \
					output_types=(tf.string,tf.string), output_shapes=(None,None))
		return_dataset = return_dataset+tuple(train_dataset)

	if val_count:
		val_generator = datagen.build_val_generator
		val_dataset = tf.data.Dataset.from_generator(val_generator, \
					output_types=(tf.string,tf.string), output_shapes=(None,None))
		return_dataset = return_dataset+tuple(val_dataset)
	return return_dataset

def dataset_Feature_Raw():

	config = setting.config
	train_dataset = tf.data.TFRecordDataset(os.path.join(config["data"]["tfrecord"]["directory"], config["data"]["tfrecord"]["train_filename"]))
	val_dataset = tf.data.TFRecordDataset(os.path.join(config["data"]["tfrecord"]["directory"], config["data"]["tfrecord"]["val_filename"]))
	return train_dataset, val_dataset

def parse_map(example_proto):
	with tf.name_scope("parsing_train_val_samples"):
		features = {"height": tf.FixedLenFeature((), tf.int64),
					"width": tf.FixedLenFeature((), tf.int64),
					"label": tf.FixedLenFeature((), tf.string),
					"image": tf.FixedLenFeature((), tf.string),
					"one_hot_label": tf.FixedLenFeature((), tf.string)}

		parsed_features = tf.parse_single_example(example_proto, features)
		height = parsed_features["height"]
		height = tf.cast(height, dtype=tf.float32)
		width = parsed_features["width"]
		width = tf.cast(width, dtype=tf.float32)

		image = parsed_features["image"]
		image = tf.image.decode_png(image, channels=3)

		label = parsed_features["label"]
		label = tf.image.decode_png(label, channels=3)

		one_hot_label = parsed_features["one_hot_label"]
		one_hot_label = tf.decode_raw(one_hot_label, tf.uint8)

		one_hot_label = tf.reshape(one_hot_label, tf.cast([tf.cast(height, tf.int64), tf.cast(width, tf.int64), 6], dtype=tf.int64))
		# image=tf.image.resize_images(image, setting.data_size, align_corners=True)
		# one_hot_label=tf.image.resize_images(one_hot_label, setting.data_size, align_corners=True)
		return tf.cast(image, tf.float32), tf.cast(one_hot_label, tf.float32)

def parse_map_with_resize(example_proto):
	with tf.name_scope("parsing_train_val_samples"):
		features = {"height": tf.FixedLenFeature((), tf.int64),
					"width": tf.FixedLenFeature((), tf.int64),
					"label": tf.FixedLenFeature((), tf.string),
					"image": tf.FixedLenFeature((), tf.string),
					"one_hot_label": tf.FixedLenFeature((), tf.string)}

		parsed_features = tf.parse_single_example(example_proto, features)
		resize_resolution = setting.data_size
		resize_height = resize_resolution[0]
		resize_width = resize_resolution[1]
		height = parsed_features["height"]
		width = parsed_features["width"]
		image = parsed_features["image"]
		image = tf.image.decode_png(image, channels=3)
		label = parsed_features["label"]
		label = tf.image.decode_png(label, channels=3)
		one_hot_label = parsed_features["one_hot_label"]
		one_hot_label = tf.decode_raw(one_hot_label, tf.uint8)
		one_hot_label = tf.reshape(one_hot_label, tf.cast([tf.cast(height, tf.int64), tf.cast(width, tf.int64), 6], dtype=tf.int64))
		# one_hot_label=tf.image.resize_images(one_hot_label, setting.data_size, align_corners=True)

		# image=tf.image.resize_images(image, setting.data_size, align_corners=True)
		return tf.cast(image, tf.float32), tf.cast(one_hot_label, tf.float32)

def parse_map_test(example_proto):

	with tf.name_scope("parsing_test_samples"):
		features = {
					"image": tf.FixedLenFeature((), tf.string, default_value="")
					}
		parsed_features = tf.parse_single_example(example_proto, features)
		image = parsed_features["image"]
		image = tf.image.decode_png(image, channels=3)
		return tf.cast(image, tf.float32), tf.cast(tf.concat([image, image], axis=-1), dtype=tf.float32)

def main():

	sess = setting.sess
	config = setting.config

	if config["phase"] == "prepare_data":
		tfRecGen = TFRecordGenerator(config["data"]["directory"]+"/train", config["data"]["directory"]+"/train_labels")
		data_file = os.path.join(config["data"]["tfrecord"]["directory"], config["data"]["tfrecord"]["train_filename"])

		if (not os.path.isfile(data_file)) or (os.stat(data_file).st_size==0):
			tfRecGen.write_TFRecord(config["data"]["tfrecord"]["train_filename"], flag="training")
		else:
			utils.print_log("training tfrecord already exists", "INFO")
			tfRecGen.read_TFRecord(config["data"]["tfrecord"]["train_filename"])

		tfRecGen = TFRecordGenerator(config["data"]["directory"]+"/val", config["data"]["directory"]+"/val_labels")
		data_file = os.path.join(config["data"]["tfrecord"]["directory"], config["data"]["tfrecord"]["val_filename"])

		if (not os.path.isfile(data_file)) or (os.stat(data_file).st_size==0):
			tfRecGen.write_TFRecord(config["data"]["tfrecord"]["val_filename"], flag="validation")
		else:
			utils.print_log("validation tfrecord already exists", "INFO")
			tfRecGen.read_TFRecord(config["data"]["tfrecord"]["val_filename"])

	elif config["phase"] == "train":
		ckpt_dir = strftime("run_%Y_%m_%d_%H:%M:%S", localtime())
		suffix_name = config["data"]["tfrecord"]["train_filename"]
		suffix_name = suffix_name[suffix_name.find("_")+1:suffix_name.find(".")]+"_original_split"
		ckpt_dir = ckpt_dir+"_"+suffix_name

		if not os.path.isdir(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir):
			os.makedirs(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir)

		if not os.path.isdir(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/best_checkpoint"):
			os.makedirs(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/best_checkpoint")

		summary_events_file = config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir
		log_file = config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/training.log"

		utils.print_log((config["training_setting"]["logging"]["training_note"]).upper(), "BOLD", log_file)

		epochs = int(config["training_setting"]["epochs"])

		model = model_hub.get_model(name=config["model"])

		next_data = namedtuple("next_data", "input label")
		with tf.name_scope("input") as scope:

			# is_training = tf.placeholder(dtype=tf.bool, shape=[], name="is_training")
			phase = tf.placeholder(tf.string, shape=[], name="phase")

			# train_dataset, val_dataset, _ = dataset_Feature_Raw()
			train_dataset, val_dataset = dataset_Feature_Raw()

			batch_size = int(config["training_setting"]["batch_size"])

			if batch_size == 1:
				utils.print_log("detected batch size: {}\noriginal image resolution will be used for training\n".upper().format(batch_size), "BOLD", log_file)
				train_dataset = train_dataset.map(parse_map)
				val_dataset = val_dataset.map(parse_map)
			else:
				utils.print_log("found batch size: {}\noptimal image resolution calculated for each batch is: {}\n".upper().format(batch_size, setting.data_size), "BOLD", log_file)
				train_dataset = train_dataset.map(parse_map_with_resize)
				val_dataset = val_dataset.map(parse_map_with_resize)

			train_dataset = train_dataset.batch(batch_size)
			train_dataset = train_dataset.shuffle(buffer_size=10000)
			train_dataset = train_dataset.repeat()
			val_dataset = val_dataset.batch(batch_size)
			handle = tf.placeholder(tf.string, shape=[], name="handle")
			iter = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,\
														train_dataset.output_shapes)

			next = iter.get_next()
			current_data = next_data(next[0], next[1])

			is_training = tf.case({tf.equal(phase, tf.constant("train")): lambda: True, tf.equal(phase, tf.constant("val")): lambda: False, tf.equal(phase, tf.constant("test")): lambda: False}, exclusive=True)
			train_iterator = train_dataset.make_one_shot_iterator()
			val_iterator = val_dataset.make_initializable_iterator()

		# with tf.name_scope("output") as scope:
		# logits = model(current_data.input, is_training)
		logits = model._setup(current_data.input, is_training)


		with tf.name_scope("performance_metrics") as scope:
			m_labels = tf.argmax(current_data.label, axis=-1)
			m_logits = tf.nn.softmax(logits)
			m_logits = tf.argmax(m_logits, axis=-1)

			false_positive, false_positive_update = tf.metrics.false_positives(current_data.label, logits, name="false_positive")
			false_negative, false_negative_update = tf.metrics.false_negatives(current_data.label, logits, name="false_negative")
			true_positive, true_positive_update = tf.metrics.true_positives(current_data.label, logits, name="true_positive")
			true_negative, true_negative_update = tf.metrics.true_negatives(current_data.label, logits, name="true_negative")
			recall, recall_update = tf.metrics.recall(labels=m_labels, predictions=m_logits, name="recall")
			mean_iou, mean_iou_update = tf.metrics.mean_iou(labels=m_labels, predictions=m_logits, num_classes=setting.num_classes, name="mean_iou")
			accuracy, accuracy_update = tf.metrics.accuracy(labels=m_labels, predictions=m_logits, name="accuracy")

			running_vars_false_positive = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="false_positive")
			running_vars_false_negative = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="false_negative")
			running_vars_true_positive = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="true_positive")
			running_vars_true_negative = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="true_negative")
			running_vars_recall = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall")
			running_vars_mean_iou = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="mean_iou")
			running_vars_accuracy = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")

			running_vars_initializer_false_positive = tf.variables_initializer(var_list=running_vars_false_positive)
			running_vars_initializer_false_negative = tf.variables_initializer(var_list=running_vars_false_negative)
			running_vars_initializer_true_positive = tf.variables_initializer(var_list=running_vars_true_positive)
			running_vars_initializer_true_negative = tf.variables_initializer(var_list=running_vars_true_negative)
			running_vars_initializer_recall = tf.variables_initializer(var_list=running_vars_recall)
			running_vars_initializer_mean_iou = tf.variables_initializer(var_list=running_vars_mean_iou)
			running_vars_initializer_accuracy = tf.variables_initializer(var_list=running_vars_accuracy)

			false_positive_rate = tf.div(false_positive, tf.add(false_positive, true_negative))
			false_negative_rate = tf.div(false_negative, tf.add(true_positive, false_negative))
			precision = tf.div(true_positive, tf.add(true_positive, false_positive))
			f1_score = tf.div(tf.multiply(2.0, tf.multiply(precision, recall)), tf.add(precision, recall))

			tf.summary.scalar("mean iou", mean_iou)
			tf.summary.scalar("accuracy", accuracy)
			tf.summary.scalar("false positive rate", false_positive_rate)
			tf.summary.scalar("false negative rate", false_negative_rate)
			tf.summary.scalar("precision", precision)
			tf.summary.scalar("f1 score", f1_score)

		with tf.name_scope("softmax_operation") as scope:
			soft_inp = tf.placeholder(name="softmax_input", shape=[None, None, None], dtype=tf.float32)
			soft_out = tf.nn.softmax(soft_inp)

		with tf.name_scope("loss") as scope:
			if config["training_setting"]["loss"] == "cross_entropy":
				no_gradient_label = tf.stop_gradient(current_data.label)
				if config["training_setting"]["class_balancing"]=="none":
					utils.print_log("No class balancing selected", "BOLD", log_file)
					losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=no_gradient_label)
				else:
					utils.print_log("Class balancing selected", "BOLD", log_file)
					weights = current_data.label * class_weights
					weights = tf.reduce_sum(weights, -1)
					losses = tf.losses.softmax_cross_entropy(onehot_labels=current_data.label, logits=logits, weights=weights)
			elif config["training_setting"]["loss"] == "lovasz":
				if not config["training_setting"]["class_balancing"]=="none":
					utils.print_log("No class balancing selected", "BOLD", log_file)
					losses = helpers.lovasz_softmax(probas=logits, labels=labels)
				else:
					utils.print_log("Class balancing selected", "BOLD", log_file)
					weights = current_data.label * class_weights
					weights = tf.reduce_sum(weights, -1)
					losses = helpers.lovasz_softmax(probas=logits, labels=labels)

			loss = tf.reduce_mean(losses)
			tf.summary.scalar("loss", loss)

		with tf.name_scope("optimization") as scope:
			if config["training_setting"]["analyse_lr"]:
				lr = tf.placeholder(name="analyse_lr", dtype=tf.float32, shape=[])
				lr_mult = float(config["training_setting"]["lr_mult"])
				lr_mult_bias = float(config["training_setting"]["lr_mult_bias"])
				optimizer = tf.train.AdamOptimizer(lr)
				# optimizer_except_bias = tf.train.AdamOptimizer(lr_mult*lr)
				# optimizer_bias = tf.train.AdamOptimizer(lr_mult_bias*lr)
			else:
				lr = float(config["training_setting"]["learning_rate"])
				lr_mult = float(config["training_setting"]["lr_mult"])
				lr_mult_bias = float(config["training_setting"]["lr_mult_bias"])

				optimizer = tf.train.AdamOptimizer(lr)
				# optimizer_except_bias = tf.train.AdamOptimizer(lr_mult*lr)
				# optimizer_bias = tf.train.AdamOptimizer(lr_mult_bias*lr)

			all_var_list = [var for var in tf.trainable_variables()]
			bias_var_list = [var for var in tf.trainable_variables() if "bias" in var.name]
			except_bias_list = list(set(all_var_list).difference(bias_var_list))

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = optimizer.minimize(loss, var_list=[var for var in tf.trainable_variables()])
				# optimize_except_bias_train_op = optimizer_except_bias.minimize(loss, var_list=except_bias_list)
				# optimize_bias_train_op = optimizer_bias.minimize(loss, var_list=bias_var_list)
				# train_op = tf.group(optimize_except_bias_train_op, optimize_bias_train_op)


			# if config["training_setting"]["clip_norm"]:
			# 	print("clip gradients will be used")
			# 	print(float(config["training_setting"]["clip_norm"]))
			# 	clip_norm = float(config["training_setting"]["clip_norm"])
			# 	optimizer = tf_utils.OptimizerCustomOperation(optimizer)
			# 	with tf.control_dependencies(update_ops):
			# 		train_op = optimizer.minimize(loss, var_list=[var for var in tf.trainable_variables()])
			# else:
			# 	print("clip gradients will not be used")
			# 	with tf.control_dependencies(update_ops):
			# 		train_op = optimizer.minimize(loss, var_list=[var for var in tf.trainable_variables()])

		with tf.name_scope("aggregate_summaries") as scope:
			merge_summary = tf.summary.merge_all()

		if not config["training_setting"]["analyse_lr"]:
			summary_writer = tf.summary.FileWriter(summary_events_file, graph=tf.get_default_graph())

		train_handle = sess.run(train_iterator.string_handle())
		val_handle = sess.run(val_iterator.string_handle())

		saver = tf.train.Saver(max_to_keep=1)
		if not config["training_setting"]["analyse_lr"]:
			meta_graph_def = tf.train.export_meta_graph(filename="./model_graphs/adapnet.meta")

		ckpt_prefix = config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/"+config["training_setting"]["checkpoints"]["prefix"]
		best_ckpt_dir = config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/best_checkpoint/"+config["training_setting"]["checkpoints"]["prefix"]

		utils.print_log("Training in progress . . . . .", "INFO", log_file)
		utils.print_log("Training start time:","INFO",log_file)
		utils.print_log(strftime("%Y_%m_%d_%H:%M:%S", localtime()),"INFO",log_file)

		global_counter = 0
		cnt = 0
		best_miou = 0.0
		early_stopping_counter = 0
		total_steps = epochs*int(train_count/batch_size)

		utils.print_log("Total steps for training: {}".format(str(total_steps)), "INFO", log_file)

		fig, ax = plt.subplots()

		# lr_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
		# lr_range = [ 0.001, 0.003, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
		lr_range = [0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.003, 0.001]
		loss_value = []

		if config["training_setting"]["analyse_lr"]:
			print("tracking loss versus learning rate curve....")
			for current_lr in lr_range:
				print("running for learning rate: {}".format(current_lr))
				print("")
				sess.run(tf.global_variables_initializer())
				i=0
				for index in range(epochs):
					try:
						for train_index in range(int(train_count/batch_size)):
							_, current_loss= sess.run([train_op, loss], feed_dict={handle: train_handle, phase:"train", lr:current_lr})
							print("step: {} loss: {}".format(i, current_loss))
							# time.sleep(1)

							i+=1
							# print(i)

					except tf.errors.OutOfRangeError as e:
						pass

				loss_value.append(current_loss)

			ax.plot(lr_range, loss_value)
			ax.set(xlabel='learning rate', ylabel='loss', title='Track best learning rate')
			ax.grid()
			fig.savefig("learning_rate_vs_loss_curve.png")
			with open("lr_vs_loss.txt", "w+") as lr_loss:
				lr_loss.write("learning rate: "+str(lr_range)+"\n")
				lr_loss.write("loss: "+str(loss_value))

		else:

			sess.run(tf.global_variables_initializer())

			for index in range(epochs):
				st = time.time()
				epoch_nbr = index
				# print(index)
				try:
					for train_index in range(int(train_count/batch_size)):

						if (cnt == (total_steps-1)*batch_size) or ((cnt-batch_size) % int(config["training_setting"]["logging"]["display_iteration"]) == 0):
							_, current_loss, cur_data = sess.run([train_op, loss, current_data], feed_dict={handle: train_handle, phase:"train"})
							message = "Epoch = %d Processed examples = %d Current_Loss = %.4f Time = %.2f"%(epoch_nbr,(cnt+batch_size),current_loss,time.time()-st)
							utils.print_log("\n"+message, "TRAINING_STATUS", log_file)
							utils.print_log("Total examples for training remaining: {}".format(str(total_steps*batch_size-(cnt+batch_size))), "INFO", log_file)
							st = time.time()

						else:
							sess.run(train_op, feed_dict={handle: train_handle, phase:"train"})

						if cnt>0 and cnt % int(config["training_setting"]["logging"]["evaluation_iteration"]) == 0:

							if not os.path.isdir(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/overlayed_images"):
								os.makedirs(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/overlayed_images")

							if not os.path.isdir(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/segmentation_images"):
								os.makedirs(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/segmentation_images")

							if not os.path.isdir(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/confidence_segmentation"):
								os.makedirs(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/confidence_segmentation")

							sess.run(val_iterator.initializer)
							# sess.run(initializers for individual metrics)

							#for initializing the local variables pertaining to each metric
							sess.run(tf.local_variables_initializer())
							for val_index in range(int(val_count/batch_size)):
								# output_logits, val_cur_data, merged_summaries, _, _, _, _, _, _, _= sess.run([logits, current_data, merge_summary, recall_update, mean_iou_update, accuracy_update, false_positive_update, false_negative_update, true_positive_update, true_negative_update], feed_dict={handle:val_handle, is_training:False})
								output_logits, val_cur_data, merged_summaries, _, _, _, _, _, _, _= sess.run([logits, current_data, merge_summary, recall_update, mean_iou_update, accuracy_update, false_positive_update, false_negative_update, true_positive_update, true_negative_update], feed_dict={handle:val_handle, phase:"val"})
								# output_logits, val_cur_data, _, _, _, _, _, _, _= sess.run([logits, current_data, recall_update, mean_iou_update, accuracy_update, false_positive_update, false_negative_update, true_positive_update, true_negative_update], feed_dict={handle:val_handle, is_training:False})
								val_index+=1
							# calculate the average metrics and display
							rec, m_iou, acc, fpr, fnr, prec, f1  = sess.run([recall, mean_iou, accuracy, false_positive_rate, false_negative_rate, precision, f1_score])
							utils.print_log("\nValidation metrics:", "VALIDATION_STATUS", log_file)
							utils.print_log("\t\trecall: {}".format(str(rec)), "VALIDATION_STATUS", log_file)
							utils.print_log("\t\tmean IoU: {}".format(str(m_iou)), "VALIDATION_STATUS", log_file)
							utils.print_log("\t\tacc: {}".format(str(acc)), "VALIDATION_STATUS", log_file)
							utils.print_log("\t\tfalse positive rate: {}".format(str(fpr)), "VALIDATION_STATUS", log_file)
							utils.print_log("\t\tfalse negative rate: {}".format(str(fnr)), "VALIDATION_STATUS", log_file)
							utils.print_log("\t\tprecision: {}".format(str(prec)), "VALIDATION_STATUS", log_file)
							utils.print_log("\t\tf1 score: {}".format(str(f1)), "VALIDATION_STATUS", log_file)

							#pick the last image from each batch for visualization
							# Plot the original prediction as segmentation image
							out_logits_sample = output_logits[-1,:,:,:]
							out_image = utils.reverse_one_hot(out_logits_sample)
							out_vis_image = utils.color_code_segmentation(out_image, setting.label_values)

							if config["training_setting"]["save_visualization_image"]:
								cv2.imwrite(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/segmentation_images/step_"+str(cnt)+".png",cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

							#use overlay visualization techniques for better visualization
							# Plot confidences as red-blue overlay
							val_image = val_cur_data.input[-1,:,:,:]
							out_logits_sample_confidence = sess.run(soft_out, feed_dict={soft_inp: out_logits_sample})
							confidence_out_vis_image = utils.make_confidence_overlay(val_image, out_logits_sample_confidence)
							
							if config["training_setting"]["save_visualization_image"]:
								scipy.misc.imsave(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/confidence_segmentation/step_"+str(cnt)+".png", confidence_out_vis_image)

							# Plot the original prediction as segmentation overlay
							overlayed_im = utils.make_overlay(np.uint8(val_image), cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
							
							if config["training_setting"]["save_visualization_image"]:
								scipy.misc.imsave(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/overlayed_images/step_"+str(cnt)+".png", overlayed_im)

							if best_miou < m_iou:
								best_miou = m_iou
								early_stopping_counter = 0
								# shutil.rmtree(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/best_checkpoint")
								tf.train.Saver().save(sess, "{0}_miou_{1:6.4f}_recall_{2:6.4f}_accuracy_{3:6.4f}_fpr_{4:6.4f}_fnr_{5:6.4f}_prec_{6:6.4f}_f1_{7:6.4f}_loss_{8:6.4f}_step_{9}".format(best_ckpt_dir, m_iou, rec, acc, fpr, fnr, prec, f1, current_loss, cnt), global_step = global_counter, write_meta_graph=False)
							# else:
							# 	early_stopping_counter+=1

							summary_writer.add_summary(merged_summaries, cnt)

						if cnt>0 and cnt % int(config["training_setting"]["checkpoints"]["save_step"]) == 0:
							saver.save(sess, ckpt_prefix, global_step = global_counter, write_meta_graph=True)
							shutil.copyfile("./config.yaml", config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/config.yaml")

						cnt = cnt + batch_size
						global_counter+=1

						# if early_stopping_counter==10:
						# 	print("Early stopping as no improvement observed for last 10 validation steps")
						# 	if not os.path.isfile(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/notes.txt"):
						# 		with open(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/notes.txt", "w+") as f:
						# 			f.write(config["training_setting"]["logging"]["training_note"])

						# 	utils.print_log("Training over......Hope I served your purpose...My Lord", "ERROR", log_file)
						# 	utils.print_log("Training end time:","INFO",log_file)
						# 	utils.print_log(strftime("%Y_%m_%d_%H:%M:%S", localtime()),"INFO",log_file)
						# 	return

						# break
					# break
				except tf.errors.OutOfRangeError as e:
					pass

			summary_writer.close()


			if not os.path.isfile(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/notes.txt"):
				with open(config["training_setting"]["checkpoints"]["save_directory"]+"/"+config["model"]+"/"+ckpt_dir+"/notes.txt", "w+") as f:
					f.write(config["training_setting"]["logging"]["training_note"])

			utils.print_log("Training over......Hope I served your purpose...My Lord", "ERROR", log_file)
			utils.print_log("Training end time:","INFO",log_file)
			utils.print_log(strftime("%Y_%m_%d_%H:%M:%S", localtime()),"INFO",log_file)

if __name__=="__main__":
    # config_dir = "./config_dir_prepare_data"
    # for config_file in sorted(os.listdir(config_dir)):
    #     tf.reset_default_graph()
    #     print("Currently running for {}".format(config_file))
    #     setting.init(config_dir+"/"+config_file)
    #     get_data_info()
    #     main()
    #     setting.sess.close()
    setting.init()
    get_data_info()
    main()
