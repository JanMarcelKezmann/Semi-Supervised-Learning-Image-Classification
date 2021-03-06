import numpy as np
import tensorflow as tf

def weight_decay(model, decay_rate, layer_name="predictions"):
	"""
	Assigns weight decay to model variables.

	Args:
		model:          tensorflow model
		decay_rate:     float value of weight decay
		layer_name:     layer to which apply weight decay, if None weight decay is applied to all layers

	Returns:
		None
	"""
	for var in model.trainable_variables:
		if layer_name:
			if layer_name in var.name:
				var.assign(var * (1 - decay_rate))
		else:
			var.assign(var * (1 - decay_rate))


def ema(model, ema_model, ema_decay):
	"""
	Updates model weights of ema model by calculating exponential moving average
	of weights of original model.

	Args:
		model:      Model from which current weights are drawn
		ema_model:  Exponential moving average model
		ema_decay:  float value for ema decay rate

	Returns:
		None
	"""
	for var, ema_var in zip(model.variables, ema_model.variables):
		if var.trainable:
			ema_var.assign((1 - ema_decay) * var + ema_decay * ema_var)
		else:
			ema_var.assign(tf.identity(var))


def linear_rampup(epochs, epoch, pre_val_iter, iteration):
	"""
	Computes linear rampup value usually used for unsupervised loss multiplier.

	Args:
		epochs:         int, number of epochs
		epoch:          int, current epoch
		pre_val_iter:   int, number of iterations prior to validation
		iteration:      int, current iteration in epoch
	
	Returns:
		float, between 0 and 1.
	"""
	total_steps = float(epochs * pre_val_iter)
	current_step = float(epoch * pre_val_iter) + (float(iteration) / float(pre_val_iter))

	rampup = np.clip(current_step / total_steps, 0., 1.)
	return float(rampup)
   