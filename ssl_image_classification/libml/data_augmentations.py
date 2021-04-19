import numpy as np
import tensorflow as tf

from multiprocessing import cpu_count

@tf.function
def weak_augment(x, height, width, pad=4, seed=[1, 2]):
    """
    Function that applies weak augmentation on batch of given images.

    Args:
        x:          tf.Tensor, batch of images of shape [batch, height, width, channels]
        height:     int, height of image
        width:      int, width of image
        pad:        int, value of paddings
        seed:       int or None, seed for random number generator

    Returns:
        Augmented input image x.
    """
	# Random right left flipping
    x = tf.image.stateless_random_flip_left_right(x, seed=seed)

    # Random Padding and Cropping
    x = tf.pad(x, paddings=[(0, 0), (pad, pad), (pad, pad), (0, 0)], mode="REFLECT")

    # Random cropping
    x = tf.map_fn(lambda batch: tf.image.stateless_random_crop(batch, size=(height, width, 3), seed=seed), x, parallel_iterations=cpu_count())

    return x


@tf.function
def medium_augment(x, height, width, pad=4, seed=[1, 2]):
    """
    Function that applies weak augmentation on batch of given images.

    Args:
        x:          tf.Tensor, batch of images of shape [batch, height, width, channels]
        height:     int, height of image
        width:      int, width of image
        pad:        int, value of paddings
        seed:       int or None, seed for random number generator

    Returns:
        Augmented input image x.
    """
    # Random right left flipping
    x = tf.image.stateless_random_flip_left_right(x, seed=seed)

    # Random brightness
    x = tf.image.stateless_random_brightness(x, max_delta=0.35, seed=seed)

    # Random contrast
    x = tf.image.stateless_random_contrast(x, lower=0, upper=0.4, seed=seed)

    # Random hue
    x = tf.image.stateless_random_hue(x, max_delta=0.2, seed=seed)

    # Random Padding and Cropping
    x = tf.pad(x, paddings=[(0, 0), (pad, pad), (pad, pad), (0, 0)], mode="REFLECT")

    # Random cropping
    x = tf.map_fn(lambda batch: tf.image.stateless_random_crop(batch, size=(height, width, 3), seed=seed), x, parallel_iterations=cpu_count())

    return x


@tf.function
def strong_augment(x, height, width, pad=4, rand_jpeg_quality=False, seed=[1, 2]):
    """
    Function that applies weak augmentation on batch of given images.

    Args:
        x:                  tf.Tensor, batch of images of shape [batch, height, width, channels]
        height:             int, height of image
        width:              int, width of image
        pad:                int, value of paddings
        rand_jpeg_quality:  Boolean, if True random JPEG quality transformation will be applied
        seed:               int or None, seed for random number generator

    Returns:
        Augmented input image x.
    """
    # Random right left flipping
    x = tf.image.stateless_random_flip_left_right(x, seed=seed)

    # Random brightness
    x = tf.image.stateless_random_brightness(x, max_delta=0.75, seed=seed)

    # Random contrast
    x = tf.image.stateless_random_contrast(x, lower=0, upper=0.8, seed=seed)

    # Random hue
    x = tf.image.stateless_random_hue(x, max_delta=0.3, seed=seed)

    # Random saturation
    x = tf.image.stateless_random_saturation(x, lower=0.5, upper=2, seed=seed)

    # Random JPEG Quality (especially for larger images)
    if rand_jpeg_quality:
        x = tf.image.stateless_random_jpeg_quality(x, min_jpeg_quality=50, max_jpeg_quality=100, seed=seed)

    # Random Padding and Cropping
    x = tf.pad(x, paddings=[(0, 0), (pad, pad), (pad, pad), (0, 0)], mode="REFLECT")

    # Random cropping
    x = tf.map_fn(lambda batch: tf.image.stateless_random_crop(batch, size=(height, width, 3), seed=seed), x, parallel_iterations=cpu_count())

    return x


def random_rotate(x):
    """
    Devides batch of images into four equally large smaller batches and rotates
    each batch by either 0, 90, 180 or 270 degrees.

    Args:
        x:  tensor, batch of input image of shape [batch, height, width, channels]

    Returns:
        Batch of rotated images and labels denoting degree of rotation, i.e. 0 
        for 0, 1 for 90, 2 for 180 and 3 for 270 degrees.
    """
    b4 = x.shape[0] // 4
    l = np.zeros(b4, np.int32)
    l = tf.constant(np.concatenate([l, l + 1, l + 2, l + 3], axis=0))
    return tf.concat([x[:b4], tf.image.rot90(x[b4:2 * b4], k=1), tf.image.rot90(x[2 * b4:3 * b4], k=2), tf.image.rot90(x[3 * b4:], k=3)], axis=0), l