import os
import numpy as np
import cv2
import random

class DataAugmentor():
	def __init__(self, input_image, label_image):

		self.input_image = input_image
		self.label_image = label_image

	def if_random(self):

		return random.randint(0,1)

	def horizontal_flip(self):

		self.input_image = cv2.flip(self.input_image,1)
		self.label_image = cv2.flip(self.label_image,1)
		return self
	    
	def vertical_flip(self):

		self.input_image = cv2.flip(self.input_image,0)
		self.label_image = cv2.flip(self.label_image,0)
		return self

	def brightness(self):

		factor = 0
		while (factor==0):
			factor = random.uniform(-1, 1)

		table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
		self.input_image = cv2.LUT(self.input_image, table)
		return self

	def rotation(self):

		angle = 0
		while (angle==0):
			angle = random.uniform(-1, 1)
    	
		angle = random.uniform(-1, 1)
		M = cv2.getRotationMatrix2D((self.input_image.shape[1]//2, self.input_image.shape[0]//2), angle, 1.0)
		self.input_image = cv2.warpAffine(self.input_image, M, (self.input_image.shape[1], self.input_image.shape[0]), flags=cv2.INTER_NEAREST)
		self.label_image = cv2.warpAffine(self.label_image, M, (self.label_image.shape[1], self.label_image.shape[0]), flags=cv2.INTER_NEAREST)
		return self

	def random_augmentation(self):
		
		self_copy = self

		if self.if_random():
			self_copy = self_copy.horizontal_flip()
		if self.if_random():
			self_copy = self_copy.vertical_flip()
		if self.if_random():
			self_copy = self_copy.brightness()
		if self.if_random():
			self_copy = self_copy.rotation()
		
		return self_copy

	def all_augmentation(self):

		return self.horizontal_flip().vertical_flip().brightness().rotation()

	def random_crop(crop_height, crop_width):

	    if (self.input_image.shape[0] != self.label_image.shape[0]) or (self.input_image.shape[1] != self.label_image.shape[1]):
	        raise Exception('Image and label must have the same dimensions!')
	        
	    if (crop_width <= self.input_image.shape[1]) and (crop_height <= self.input_image.shape[0]):
	        x = random.randint(0, self.input_image.shape[1]-crop_width)
	        y = random.randint(0, self.input_image.shape[0]-crop_height)
	        
	        if len(self.label_image.shape) == 3:
	            return self.input_image[y:y+crop_height, x:x+crop_width, :], self.label_image[y:y+crop_height, x:x+crop_width, :]
	        else:
	            return self.input_image[y:y+crop_height, x:x+crop_width, :], self.label_image[y:y+crop_height, x:x+crop_width]
	    else:
	        raise Exception('Crop shape exceeds image dimensions!')


