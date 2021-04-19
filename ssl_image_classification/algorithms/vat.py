import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def kl_divergence_from_logits(logits_a, logits_b):
	"""
	Gets KL divergence from logits parameterizing categorical distributions.

	Args:
		logits_a:   tensor, model outputs of input a
		logits_b:   tensor, model outputs of input b

	Returns:
		Tensor with the batchwise KL-divergence between distrib_a and distrib_b.

	"""
	# Compute categorical distribution over logits
	distrib_a = tfp.distributions.Categorical(logits=logits_a)
	distrib_b = tfp.distributions.Categorical(logits=logits_b)

	return tfp.distributions.kl_divergence(distrib_a, distrib_b)


@tf.function
def entropy(logits):
	"""
	Computes Entropy of model outputs, i.e. logits.

	Args:
		logits:     tensor, model outputs

	Returns:
		Tensor, holding Shannon Entropy of model outputs
	"""
	return tfp.distributions.Categorical(logits=logits).entropy()


@tf.function
def kl_divergence_with_logits(logits_a, logits_b):
	"""
	Compute the per-element KL-divergence of a batch.

	Args:
		logits_a:   tensor, model outputs of input a
		logits_b:   tensor, model outputs of input b

	Returns:
		Tensor of per-element KL-divergence of model outputs a and b
	"""
	a = tf.nn.softmax(logits_a, axis=1)
	a_loga = tf.reduce_sum(a * log_softmax(logits_a), 1)
	a_logb = tf.reduce_sum(a * log_softmax(logits_b), 1)

	return a_loga - a_logb


@tf.function
def get_normalized_vector(v):
	"""
	Normalize v by infinity and L2 norms.

	Args:
		v:  tensor, here tensor holding values of given distributions

	Returns:
		Normalized input by infinity and L2 norms.
	"""
	v /= 1e-12 + tf.reduce_max(
		tf.abs(v), list(range(1, len(v.get_shape()))), keepdims=True
	)
	v /= tf.sqrt(
		1e-6 + tf.reduce_sum(
			tf.pow(v, 2.0), list(range(1, len(v.get_shape()))), keepdims=True
		)
	)
	
	return v


@tf.function
def log_softmax(x):
	"""
	Compute log-domain softmax of logits

	Args:
		x:  tensor, here logits

	Returns:
		tensor, log-domain softmax of input
	"""
	x_dev = x - tf.reduce_max(x, 1, keepdims=True)
	logsoftmax = x_dev - tf.math.log(tf.reduce_sum(tf.exp(x_dev), 1, keepdims=True))

	return logsoftmax


@tf.function
def vat(x, logits, model, v, eps, xi=1e-6):
	"""
	Generate an adeversarial perturbation.

	Args:
		x:          tensor, batch of labeled input images of shape [batch, height, width, channels]
		logits:     tensor, holding model outputs of input
		model:      tf.keras model
		v:          generator, random number generator
		eps:        float, small epsilon
		xi:         float, small xi

	Returns:
		Adversarial perturbation to be applied to x.
	"""
	# v = tf.random.Generator.from_non_deterministic_state()
	# v = tf.random.normal(shape=tf.shape(x))

	v = xi * get_normalized_vector(v.normal(shape=tf.shape(x)))
	logits_p = logits
	logits_m = model(x + v, training=True)[0]
	dist = kl_divergence_with_logits(logits_p, logits_m)
	grad = tf.gradients(tf.reduce_mean(dist), [v], aggregation_method=2)[0]
	v = tf.stop_gradient(grad)

	return eps * get_normalized_vector(v)


@tf.function
def ssl_loss_vat(labels_x, logits_x, logits_student, logits_teacher, logits_u):
	"""
	Computes cross entropy loss based on the labeled data model outputs, a
	vat KL loss on the unlabeled data model outputs and its guessed teacher
	logits and entropy loss on unlabeled input

	Args:
		labels_x:   tensor, contains labels corresponding to logits_x of shape [batch, num_classes]
		logits_x:   tensor, contains the logits of an batch of images of shape [batch, num_classes]
		labels_u:   tensor, contains labels corresponding to logits_u of shape [batch, num_classes]
		logits_u:   tensor, contains the logits of an batch of images of shape [batch, num_classes]

	Returns:
		Three floating point numbers, the first holding the labeled CE loss, the 
		second holding the VAT KL divergence loss of the student and teacher
		model outputs and the third holding the shannon entropy of the unlabeled
		images.
	"""
	x_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
	x_loss = tf.reduce_mean(x_loss)

	loss_vat = kl_divergence_from_logits(logits_a=logits_student, logits_b=logits_teacher)
	loss_vat = tf.reduce_mean(loss_vat)

	loss_entropy = entropy(logits=logits_u)
	loss_entropy = tf.reduce_mean(loss_entropy)

	return x_loss, loss_vat, loss_entropy