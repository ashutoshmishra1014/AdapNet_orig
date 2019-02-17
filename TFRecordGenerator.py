import numpy as np
import os
import tensorflow as tf
import cv2
import sys
from progressbar import ProgressBar
import time
from PIL import Image
from DataAugmentor import DataAugmentor
import setting
import utils



tfr_dir = "./tf_datasets"


class ImageCoder(object):
	"""Helper class that provides TensorFlow image coding utilities."""

	def __init__(self):
		# Create a single Session to run all image coding calls.
		# self._sess = tf.Session()
		self._sess = setting.sess
		# Initializes function that converts PNG to JPEG data.
		self._png_data = tf.placeholder(dtype=tf.string)
		self._raw_png_data = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)
		self._raw_jpg_data = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)

		image = tf.image.decode_png(self._png_data, channels=3)
		self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
		# Initializes function that decodes RGB JPEG data.
		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

		self._encode_png = tf.image.encode_png(self._raw_png_data)
		self._encode_jpeg = tf.image.encode_jpeg(self._raw_jpg_data, format='rgb', quality=100)

	def png_to_jpeg(self, image_data):

		return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

	def decode_jpeg(self, image_data):

		image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

	def encode_png(self, image_data):

		return self._sess.run(self._encode_png,
                          feed_dict={self._raw_png_data: image_data})

	def encode_jpeg(self, image_data, format=format):

		return self._sess.run(self._encode_jpeg,
                          feed_dict={self._raw_jpg_data: image_data})

	def is_png(self, filename):

		return (os.path.splitext(filename)[1]==".png") or (os.path.splitext(filename)[1]==".PNG")

	def is_jpg(self, filename):

		return (os.path.splitext(filename)[1]==".jpg") or (os.path.splitext(filename)[1]==".jpeg") \
				(os.path.splitext(filename)[1]==".JPG") or (os.path.splitext(filename)[1]==".JPEG")

class TFRecordGenerator():

	def __init__(self, image_dir, label_dir):
		self.image_dir = image_dir
		self.label_dir = label_dir
		self.coder = ImageCoder()
		self.augmentor = None
		self.label_values = setting.label_values
		self.sess = setting.sess

	def return_addrs(self, file_dir):

		addrs = []
		for fn in sorted(os.listdir(file_dir)):
			addrs.append(os.path.join(file_dir,fn))
		return addrs

	def generate_augmented_data(self, image, label):
		augmentor = DataAugmentor(image, label)
		if setting.config["data_processing"]["all_augmentation"]:
			augmentor = augmentor.all_augmentation()
		elif setting.config["data_processing"]["random_augmentation"]:
			augmentor = augmentor.random_augmentation()
		else:
			if setting.config["data_processing"]["horizontal_flip"]:
				augmentor = augmentor.horizontal_flip()
			if setting.config["data_processing"]["vertical_flip"]:
				augmentor = augmentor.vertical_flip()
			if setting.config["data_processing"]["brightness"]:
				augmentor = augmentor.brightness()
			if setting.config["data_processing"]["rotation"]:
				augmentor = augmentor.rotation()

		return augmentor.input_image, augmentor.label_image



	def process_image(self, image_filename, label_filename):
		#not using self.coder.png_to_jpeg coz, enode_png compressed data relatively more 
		image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
		label_data = tf.gfile.FastGFile(label_filename, 'rb').read()
		image = cv2.imread(image_filename)
		label = cv2.imread(label_filename)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
		#augmentation logic goes here, handling all the possible cases
		if (setting.config["data_processing"]["horizontal_flip"] or setting.config["data_processing"]["vertical_flip"] or setting.config["data_processing"]["brightness"] or setting.config["data_processing"]["rotation"] \
			or setting.config["data_processing"]["all_augmentation"] or setting.config["data_processing"]["random_augmentation"]):
			image, label = self.generate_augmented_data(image, label)
		
		height, width, _ = image.shape
		height = tf.cast(height, dtype=tf.float32)
		width = tf.cast(width, dtype=tf.float32)

		# check_height_even_odd = tf.cast(tf.mod(tf.ceil(tf.div(height, 8)), 2), dtype=tf.int32)
		# check_width_even_odd = tf.cast(tf.mod(tf.ceil(tf.div(width, 8)), 2), dtype=tf.int32)

		# new_height = tf.cond(tf.equal(check_height_even_odd, 0), lambda: tf.cast(height, dtype=tf.int32), lambda: tf.cast((tf.ceil(tf.div(height, 8))-1)*8, dtype=tf.int32))
		# new_width = tf.cond(tf.equal(check_width_even_odd, 0), lambda: tf.cast(width, dtype=tf.int32), lambda: tf.cast((tf.ceil(tf.div(width, 8))-1)*8, dtype=tf.int32))
		
		# image=self.sess.run(tf.image.resize_images(image, [new_height, new_width], align_corners=True))
		# label=self.sess.run(tf.image.resize_images(image, [new_height, new_width], align_corners=True))


		one_hot_label = utils.one_hot_it(label, self.label_values)
		one_hot_label = one_hot_label.tobytes()

		if self.coder.is_png(image_filename):
			image_data = self.coder.encode_png(image)
			# image_data = self.coder.png_to_jpeg(image_data)
		elif self.coder.is_jpg(image_filename):
			image_data = self.coder.encode_jpeg(image, format="rgb")

		if self.coder.is_png(label_filename):
			label_data = self.coder.encode_png(label)
			# image_data = self.coder.png_to_jpeg(image_data)
		elif self.coder.is_jpg(label_filename):
			label_data = self.coder.encode_jpeg(label, format="rgb")

		image = self.coder.decode_jpeg(image_data)
		assert len(image.shape) == len(label.shape) == 3
		height = image.shape[0]
		width = image.shape[1]
		assert image.shape[2] == 3


		return image_data, label_data, height, width, one_hot_label

	def process_image_test(self, image_filename):
		#not using self.coder.png_to_jpeg coz, enode_png compressed data relatively more 
		image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
		image = cv2.imread(image_filename)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


		height, width, _ = image.shape
		height = tf.cast(height, dtype=tf.float32)
		width = tf.cast(width, dtype=tf.float32)

		# check_height_even_odd = tf.cast(tf.mod(tf.ceil(tf.div(height, 8)), 2), dtype=tf.int32)
		# check_width_even_odd = tf.cast(tf.mod(tf.ceil(tf.div(width, 8)), 2), dtype=tf.int32)

		# new_height = tf.cond(tf.equal(check_height_even_odd, 0), lambda: tf.cast(height, dtype=tf.int32), lambda: tf.cast((tf.ceil(tf.div(height, 8))-1)*8, dtype=tf.int32))
		# new_width = tf.cond(tf.equal(check_width_even_odd, 0), lambda: tf.cast(width, dtype=tf.int32), lambda: tf.cast((tf.ceil(tf.div(width, 8))-1)*8, dtype=tf.int32))

		# image=self.sess.run(tf.image.resize_images(image, [new_height, new_width], align_corners=True))


		if self.coder.is_png(image_filename):
			image_data = self.coder.encode_png(image)
			# image_data = self.coder.png_to_jpeg(image_data)
		elif self.coder.is_jpg(image_filename):
			image_data = self.coder.encode_jpeg(image, format="rgb")

		image = self.coder.decode_jpeg(image_data)

		return image_data

	def load_image(self, image, label):

	    image = cv2.imread(image)
	    label = cv2.imread(label)
	    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
	    #image, label = generate_augmented_data(image, label)
	    image = tf.image.encode_png(image)
	    label = tf.image.encode_png(label)

	    return sess.run(image), sess.run(label)

	def int64_feature(self, value):

		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def bytes_feature(self, value):

		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def float_feature(self, value):

		return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

	def write_TFRecord(self, filename, flag=None):

		trackProgress = ProgressBar()
		writer = tf.python_io.TFRecordWriter(os.path.join(setting.config["data"]["tfrecord"]["directory"],filename))
		image_addrs = self.return_addrs(self.image_dir)
		label_addrs = self.return_addrs(self.label_dir)
		utils.print_log("Proceeding with writing tfrecords for "+flag, "INFO")
		for i in trackProgress(range(len(image_addrs))):
			# image, label = self.load_image(image_addrs[i], label_addrs[i])
			image, label, height, width, one_hot_label = self.process_image(image_addrs[i], label_addrs[i])
			# label, _, __ = self.process_image(label_addrs[i])


			feature = {
			'height':self.int64_feature(height),
			'width':self.int64_feature(width),
			'label': self.bytes_feature(label),
			'image': self.bytes_feature(image),
			'one_hot_label': self.bytes_feature(one_hot_label)
			}
			example = tf.train.Example(features=tf.train.Features(feature=feature))

			writer.write(example.SerializeToString())
		    
		writer.close()
		sys.stdout.flush()
		utils.print_log("Writing tfrecords is completed", "INFO")

	def write_TFRecord_test(self, filename):

		trackProgress = ProgressBar()
		writer = tf.python_io.TFRecordWriter(os.path.join(setting.config["data"]["tfrecord"]["directory"],filename))
		image_addrs = self.return_addrs(self.image_dir)
		utils.print_log("Proceeding with writing tfrecords for testing", "INFO")
		for i in trackProgress(range(len(image_addrs))):
			
			image = self.process_image_test(image_addrs[i])

			feature = {
			'image': self.bytes_feature(image)
			}
			example = tf.train.Example(features=tf.train.Features(feature=feature))
			writer.write(example.SerializeToString())
		    
		writer.close()
		sys.stdout.flush()
		utils.print_log("Writing test tfrecords is completed", "INFO")

	def read_TFRecord(self, filename):
		data_path = os.path.join(setting.config["data"]["tfrecord"]["directory"],filename)  # address to save the hdf5 file
		reconstructed_images = []

		record_iterator = tf.python_io.tf_record_iterator(path=data_path)

		for each_record in record_iterator:

		    example = tf.train.Example()
		    example.ParseFromString(each_record)
		    
		    height = int(example.features.feature['height']
		                                 .int64_list
		                                 .value[0])
		    
		    width = int(example.features.feature['width']
		                                .int64_list
		                                .value[0])
		    
		    img_string = (example.features.feature['image']
		                                  .bytes_list
		                                  .value[0])
		    
		    label_string = (example.features.feature['label']
		                                .bytes_list
		                                .value[0])

		    one_hot_label_string = (example.features.feature['one_hot_label']
		                                .bytes_list
		                                .value[0])

		    reconstructed_img = tf.image.decode_png(img_string, channels=3)		    
		    reconstructed_label = tf.image.decode_png(label_string, channels=3)
		    reconstructed_one_hot_label = tf.decode_raw(one_hot_label_string, tf.int32)
		    reconstructed_one_hot_label = tf.reshape(reconstructed_one_hot_label, tf.cast([tf.cast(height, tf.int64), tf.cast(width, tf.int64), 6], dtype=tf.int64))
		    #display reconstructed images
		    img = Image.fromarray(setting.sess.run(reconstructed_img), "RGB")
		    img.show()
		    img = Image.fromarray(setting.sess.run(reconstructed_label), "RGB")
		    img.show()
		    reconstructed_images.append((reconstructed_img, reconstructed_label))
		    break


	def read_TFRecord_test(self, filename):
		data_path = os.path.join(setting.config["data"]["tfrecord"]["directory"],filename)  # address to save the hdf5 file
		reconstructed_images = []

		record_iterator = tf.python_io.tf_record_iterator(path=data_path)

		for each_record in record_iterator:

		    example = tf.train.Example()
		    example.ParseFromString(each_record)
		    
		    img_string = (example.features.feature['image']
		                                  .bytes_list
		                                  .value[0])

		    reconstructed_img = tf.image.decode_png(img_string, channels=3)		    
		    #display reconstructed images
		    img = Image.fromarray(setting.sess.run(reconstructed_img), "RGB")
		    img.show()
		    reconstructed_images.append(reconstructed_img)
		    break