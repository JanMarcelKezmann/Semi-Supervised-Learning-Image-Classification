import tensorflow as tf

def weight_decay(model, decay_rate):
	for var in model.trainable_variables:
		var.assign(var * (1 - decay_rate))


def ema(model, ema_model, ema_decay):
	for var, ema_var in zip(model.variables, ema_model.variables):
		if var.trainable:
			ema_var.assign((1 - ema_decay) * var + ema_decay * ema_var)
		else:
			ema_var.assign(tf.identity(var))
   