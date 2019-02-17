import numpy as np
import tensorflow as tf
import os
from PIL import Image
import utils

class DatasetGenerator():

	def __init__(self, data_dir):
		utils.print_log("\nInitiating run for creating data generators", "INFO")
		print("-"*50)
		print("-"*50)
		self.data_dir = data_dir
		self.train_files_list, self.train_labels_list = self.setup(data_dir, flag="train")
		self.val_files_list, self.val_labels_list = self.setup(data_dir, flag="val")
		# self.test_files_list = self.setup(data_dir, flag="test")


	def setup(self, data_dir, flag=None):
		files_list = []
		labels_list = []

		if flag=="train" or flag=="val":
			for fn in sorted(os.listdir(data_dir+"/"+flag)):
				if not os.path.isdir(data_dir+"/"+flag+"/"+fn):
					files_list.append(data_dir+"/"+flag+"/"+fn)

			for label_fn in sorted(os.listdir(data_dir+"/"+flag+"_labels")):
				if not os.path.isdir(data_dir+"/"+flag+"_labels/"+label_fn):
					labels_list.append(data_dir+"/"+flag+"_labels/"+label_fn)

			if len(files_list) != len(labels_list):
				utils.print_log("There seems to be inconsistent number of "+flag+" images and labels, please check", "ERROR")
			else:
				utils.print_log("{} images size: {}".format(flag, len(files_list)), "INFO")
				utils.print_log("{} label images size: {}".format(flag, len(labels_list)), "INFO")
				print("-"*50)

			return files_list, labels_list
		elif flag=="test":
			for fn in sorted(os.listdir(data_dir+"/"+flag)):
				if not os.path.isdir(data_dir+"/"+flag+"/"+fn):
					files_list.append(data_dir+"/"+flag+"/"+fn)

			utils.print_log("{} images size: {}".format(flag, len(files_list)), "INFO")
			print("-"*50)

			return files_list


	def build_train_generator(self):
		for i in range(len(self.train_files_list)):
			yield self.train_files_list[i], self.train_labels_list[i]

	def build_val_generator(self):
		for i in range(len(self.val_files_list)):
			yield self.val_files_list[i], self.val_labels_list[i]

	def build_test_generator(self):
		for i in range(len(self.test_files_list)):
			yield self.test_files_list[i]


