from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

from . import mixup


@tf.function
def augment(x, height, width):
	# Random right left flipping
	x = tf.image.random_flip_left_right(x)

	# Random Padding and Cropping
	x = tf.pad(x, paddings=[(0, 0), (4, 4), (4, 4), (0, 0)], mode="REFLECT")
	
	# Random cropping
	x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(height, width, 3)), x, parallel_iterations=cpu_count())
	
	return x


def guess_labels(u_aug, model, K):
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
	batch_size = x.shape[0]
	
	x_aug = augment(x, height, width)
	u_aug = [None for _ in range(K)]
	
	for k in range(K):
		u_aug[k] = augment(u, height, width)
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
	loss_x = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
	loss_x = tf.reduce_mean(loss_x)
	loss_u = tf.square(labels_u - tf.nn.softmax(logits_u))
	loss_u = tf.reduce_mean(loss_u)

	return loss_x, loss_u


def linear_rampup(epoch, rampup=16):
    if rampup == 0:
        return 1.
    else:
        rampup = np.clip(epoch / rampup, 0., 1.)
        return float(rampup)