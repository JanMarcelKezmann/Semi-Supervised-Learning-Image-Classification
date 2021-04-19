from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

from . import mixup
from ..libml.data_augmentations import weak_augment, medium_augment, strong_augment

def guess_labels(u_aug, model, K):
	"""
	Computes and returns labels for unlabeled and augmented data.

	Args:
		u_aug:  tensor, unlabeled batch of augmented images of shape [K, batch, height, width, channels]
		model:  tensorflow model
		K:      int, number of augmentations

	Returns:
		batch of labels of shape [batch, num_classes]
	"""
	u_logits = tf.nn.softmax(model(u_aug[0])[0], axis=1)
	for k in range(1, K):
		u_logits += tf.nn.softmax(model(u_aug[k])[0], axis=1)
	u_logits = u_logits / K
	
	# Stops gradient compuation
	u_logits = tf.stop_gradient(u_logits)

	return u_logits


@tf.function
def sharpen(p, T):
	return tf.pow(p, 1 / T) / tf.reduce_sum(tf.pow(p, 1 / T), axis=1, keepdims=True)


def interleave_offsets(batch, nu):
	groups = [batch // (nu + 1)] * (nu + 1)
	for x in range(batch - sum(groups)):
		groups[-x - 1] += 1

	offsets = [0]
	for g in groups:
		offsets.append(offsets[-1] + g)

	assert offsets[-1] == batch
	
	return offsets


def interleave(xy, batch):
	nu = len(xy) - 1
	offsets = interleave_offsets(batch, nu)
	xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
	for i in range(1, nu + 1):
		xy[0][i], xy[i][i] = xy[i][i], xy[0][i]

	return [tf.concat(v, axis=0) for v in xy]


def mixmatch(model, x, y, u, T, K, beta, height, width):
	"""
	Applies mixmatch algorithm on inputs x, y and u returns mixmatched tensors
	X_prime and U_prime

	Args:
		model:      tf.keras Model
		x:          tensor, labeled batch of images [batch, height, width, channels]
		y:          tensor, batch of labels of x with shape [batch, num_classes]
		u:          tensor, unlabeled batch of images [bathc, height, widht, channels]
		T:          float, sharpening temperature
		K:          int, number of augmentations
		beta:       tensor, holding beta distributed values
		height:     int, height of images
		width:      int, width of images

	Returns:
		Two tensors, one holding the transformed input images consisting of a
		transformation of the mixed labeled and unlabeled data, the other one
		holds its corresponding aggregated labels.
	"""
	batch_size = x.shape[0]

	x_aug = weak_augment(x, height, width)
	u_aug = [None for _ in range(K)]

	for k in range(K):
		u_aug[k] = weak_augment(u, height, width)
	u_aug_labels = guess_labels(u_aug, model, K)
	u_aug_labels = sharpen(u_aug_labels, tf.constant(T))

	U = tf.concat(u_aug, axis=0)
	u_aug_labels = tf.concat([u_aug_labels for _ in range(K)], axis=0)

	X_aug_U = tf.concat([x_aug, U], axis=0)
	y_u_aug_labels = tf.concat([y, u_aug_labels], axis=0)

	indices = tf.random.shuffle(tf.range(X_aug_U.shape[0]))

	W_X_aug_U = tf.gather(X_aug_U, indices)
	W_y_u_aug_labels = tf.gather(y_u_aug_labels, indices)

	X_prime, U_prime = mixup(X_aug_U, W_X_aug_U, y_u_aug_labels, W_y_u_aug_labels, beta=beta, alg="mixmatch")
	X_prime = tf.split(X_prime, K + 1, axis=0)
	X_prime = interleave(X_prime, batch_size)

	return X_prime, U_prime


@tf.function
def ssl_loss_mixmatch(labels_x, logits_x, labels_u, logits_u):
	"""
	Computes cross entropy loss based on the labeled data model outputs and the
	mean squared error based on the unlabeled data model outputs and its guessed
	labels.
	loss_x is referring to the labeled CE loss and loss_u to the unlabeled MSE.

	Args:
		labels_x:   tensor, contains labels corresponding to logits_x of shape [batch, num_classes]
		logits_x:   tensor, contains the logits of an batch of images of shape [batch, num_classes]
		labels_u:   tensor, contains labels corresponding to logits_u of shape [batch, num_classes]
		logits_u:   tensor, contains the logits of an batch of images of shape [batch, num_classes]
		
	Returns:
		Two floating point numbers, the first holding the labeled CE loss, the 
		second holding the MSE of the unlabeled images.
	"""
	loss_x = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
	loss_x = tf.reduce_mean(loss_x)
	loss_u = tf.square(labels_u - tf.nn.softmax(logits_u))
	loss_u = tf.reduce_mean(loss_u)

	return loss_x, loss_u
