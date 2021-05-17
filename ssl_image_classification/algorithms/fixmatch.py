from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf


from ..libml.data_augmentations import weak_augment, medium_augment, strong_augment


def guess_labels(x, model, uratio):
    """
    Computes and returns labels for unlabeled and augmented data.

    Args:
        x:          tensor, unlabeled batch of images of shape [uratio, batch, height, width, channels]
        model:      tensorflow model
        uratio:     int, number of augmentations

    Returns:
        batch of labels of shape [batch * uratio, num_classes]
    """
    labels_x = [tf.nn.softmax(model(x[k], training=True)[0], axis=1) for k in range(uratio)]
    labels_x = tf.concat(labels_x, axis=0)
    
    # Stops gradient compuation
    labels_x = tf.stop_gradient(labels_x)

    return labels_x


def fixmatch(model, x, y, u, height, width, uratio=2):
    """
    Applies fixmatch algorithm on inputs x, y and u returns tensors
    X_prime and U_prime and a float holding the Kullback-Leibler loss.

    Args:
        model:      tf.keras Model
        x:          tensor, labeled batch of images [batch, height, width, channels]
        y:          tensor, batch of labels of x with shape [batch, num_classes]
        u:          tensor, unlabeled batch of images [uratio * batch, height, widht, channels]
        height:     int, height of images
        width:      int, width of images
        uratio:     int, unlabeled batch size ratio

    Returns:
        Three tensors, the first holds the augemented labeled input x, the
        second the weakly augmented unlabeled images and third the strongly
        augmentend unlabeled images
    """
    batch_size = x.shape[0]

    x_aug = weak_augment(x, height, width)
    u_weak_aug = [None for _ in range(uratio)]
    u_strong_aug = [None for _ in range(uratio)]

    # Applying weak and strong augmentation to unlabeled data
    for k in range(uratio):
        u_weak_aug[k] = weak_augment(u[k * batch_size:(k + 1) * batch_size], height, width)
    for k in range(uratio):
        u_strong_aug[k] = strong_augment(u[k * batch_size:(k + 1) * batch_size], height, width)

    # guess strong labels by computing model output of same weakly augmented data
    labels_strong = guess_labels(u_weak_aug, model, uratio)

    return x_aug, labels_strong, u_strong_aug


@tf.function
def ssl_loss_fixmatch(labels_x, logits_x, labels_strong, logits_strong, confidence=0.95):
    """
    Computes cross entropy loss based on the labeled data model outputs and a
    pseudo label cross entropy loss on the unlabeled data model outputs and its
    guessed pseudo labels.
    loss_x is referring to the labeled CE loss and loss_u to the unlabeled 
    pseudo CE loss.

    Args:
        labels_x:       tensor, contains labels corresponding to logits_x of shape [batch, num_classes]
        logits_x:       tensor, contains the logits of an batch of images of shape [batch, num_classes]
        labels_strong:  tensor, contains labels corresponding to logits_u of shape [batch, num_classes]
        logits_strong:  tensor, contains the logits of an batch of images of shape [batch, num_classes]
        confidence:     float, indicates amount of certainty needed in pseudo labels to be used as
                            True labels for strongly augmented unlabeled data

    Returns:
        Two floating point numbers, the first holding the labeled CE loss, the 
        second holding the Pseudo CE loss of the unlabeled images.
    """
    # CE loss for labeled data
    loss_x = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
    loss_x = tf.reduce_mean(loss_x)

    # Pseudo-label CE for unlabeled data
    # one_hot_argmaxed_labels_strong = tf.argmax(pseudo_labels, axis=1)
    # one_hot_argmaxed_labels_strong = tf.one_hot(pseudo_labels, depth=labels_x.shape[1])
    loss_xeu = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tf.argmax(labels_strong, axis=1), depth=labels_x.shape[1]), logits=logits_strong)
    pseudo_mask = tf.cast(tf.reduce_max(labels_strong, axis=1) >= confidence, tf.float32)
    loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)

    return loss_x, loss_xeu