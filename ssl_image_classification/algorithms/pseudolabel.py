import tensorflow as tf

from ..libml.data_augmentations import weak_augment, medium_augment, strong_augment


def pseudo_label(x, u, height, width):
	"""
	Applies medium augmentations on inputs x and u returns tensors
	and returns augmented tensors.

	Args:
		x:          tensor, labeled batch of images [batch, height, width, channels]
		u:          tensor, unlabeled batch of images [batch, height, widht, channels]
		height:     int, height of images
		width:      int, width of images

	Returns:
		Augmented input tensors.
	"""
	x_aug = medium_augment(x, height, width)
	u_aug = medium_augment(u, height, width)

	return x_aug, u_aug


@tf.function
def ssl_loss_pseudo_label(labels_x, logits_x, logits_u, threshold=0.95):
	"""
	Computes two cross entropy losses based on the labeled and unlabeled data.
	loss_x is referring to the labeled CE loss and loss_u to the unlabeled CE loss.

	Args:
		labels_x:   tensor, contains labels corresponding to logits_x of shape [batch, num_classes]
		logits_x:   tensor, contains the logits of an batch of images of shape [batch, num_classes]
		logits_u:   tensor, logits of unlabeled model output of shape [batch, num_classes]
		threshold:  float, high confidence parameter
		
	Returns:
		Two floating point numbers, the first representing the labeled CE loss
		and the second holding the high confidence CE loss values.
	"""
	x_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
	x_loss = tf.reduce_mean(x_loss)

	pl_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(logits_u, axis=-1), logits=logits_u)

	# Create mask denoting which data points have high-confidence predictions
	greater_than_threshold = tf.reduce_any(tf.greater(tf.nn.softmax(logits_u, axis=1), threshold), axis=-1, keepdims=True)
	greater_than_threshold = tf.cast(greater_than_threshold, pl_loss.dtype)

	# Enforcing loss only when model is confident enough
	pl_loss = pl_loss * greater_than_threshold
	pl_loss = tf.reduce_mean(pl_loss)

	return x_loss, pl_loss