import tensorflow as tf
import numpy as np

class OptimizerCustomOperation:
	def __init__(self, opt):
		self.opt = opt

	def minimize(self, loss=None, var_list=None, clip_norm=None, global_step=None):
		grads_and_vars = self.opt.compute_gradients(loss, var_list)
		grads, tvars = zip(*grads_and_vars)
		grads, global_norm = tf.clip_by_global_norm(grads, clip_norm)
		grads_and_vars = zip(grads, tvars)
		train_op = self.opt.apply_gradients(grads_and_vars, global_step)
