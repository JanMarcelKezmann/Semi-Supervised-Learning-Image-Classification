from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf


@tf.function
def weak_augment(x, height, width):
    # Random right left flipping
    x = tf.image.random_flip_left_right(x)

    # Random Padding and Cropping
    x = tf.pad(x, paddings=[(0, 0), (4, 4), (4, 4), (0, 0)], mode="REFLECT")
    
    # Random cropping
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(height, width, 3)), x, parallel_iterations=cpu_count())
    
    return x


@tf.function
def strong_augment(x, height, width):
    # Random right left flipping
    x = tf.image.random_flip_left_right(x)

    # Random brightness
    x = tf.image.random_brightness(x, 0.25)

    # Random contrast
    x = tf.image.random_contrast(x, 0, 0.4)

    # Random hue
    x = tf.image.random_hue(x, 0.2)

    # Random saturation
    x = tf.image.random_saturation(x, 0, 3)

    # Random Padding and Cropping
    x = tf.pad(x, paddings=[(0, 0), (4, 4), (4, 4), (0, 0)], mode="REFLECT")

    # Random cropping
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(height, width, 3)), x, parallel_iterations=cpu_count())

    return x


def guess_labels(x, model, uratio):
    labels_x = [tf.nn.softmax(model(x[k], training=True)[0], axis=1) for k in range(uratio)]
    labels_x = tf.concat(labels_x, axis=0)
    
    # Stops gradient compuation
    labels_x = tf.stop_gradient(labels_x)

    return labels_x


def fixmatch(model, x, y, u, height, width, uratio=2):
    batch_size = x.shape[0]

    x_aug = weak_augment(x, height, width)
    u_weak_aug = [None for _ in range(uratio)]
    u_strong_aug = [None for _ in range(uratio)]

    for k in range(uratio):
        u_weak_aug[k] = weak_augment(u[k * batch_size:(k + 1) * batch_size], height, width)
    for k in range(uratio):
        u_strong_aug[k] = strong_augment(u[k * batch_size:(k + 1) * batch_size], height, width)

    labels_strong = guess_labels(u_weak_aug, model, uratio)

    return x_aug, labels_strong, u_strong_aug


@tf.function
def ssl_loss_fixmatch(labels_x, logits_x, labels_strong, logits_strong, confidence=0.95):
    loss_x = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
    loss_x = tf.reduce_mean(loss_x)

    # Pseudo-label CE for unlabeled data
    loss_xeu = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tf.argmax(labels_strong, axis=1), depth=labels_x.shape[1]), logits=logits_strong)
    # one_hot_argmaxed_labels_strong = tf.argmax(pseudo_labels, axis=1)
    # one_hot_argmaxed_labels_strong = tf.one_hot(pseudo_labels, depth=labels_x.shape[1])
    pseudo_mask = tf.cast(tf.reduce_max(labels_strong, axis=1) >= confidence, tf.float32)
    loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)

    return loss_x, loss_xeu