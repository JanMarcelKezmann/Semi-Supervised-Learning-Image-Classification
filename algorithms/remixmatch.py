from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

from . import mixup
from ..libml.data_augmentations import weak_augment, medium_augment, strong_augment, random_rotate



@tf.function
def compute_rot_loss(x, model, w_rot=0.5):
    """
    Compute auxiliary rotation loss.

    Args:
        x:      tensor, batch of 0, 90, 180 and 270 degrees rotated labeled images of shape [batch, height, width, channels]
        model:  tf.keras Model
        w_rot:  float, if > 0.0 rotation loss will be computed else 0 will be returned

    Returns:
        Returns either 0 if w_rot == 0.0 or the rotation loss of the input images
    """
    if w_rot > 0:
        # Create rotated batch and its corresponding labels
        y_rot, labels_rot = random_rotate(x)
        
        # Compute model output with 4 output units and apply one hot encoding to receive labels
        logits_rot = tf.keras.layers.Dense(4, kernel_initializer="glorot_normal")(model(y_rot, training=True)[1])
        labels_rot = tf.one_hot(labels_rot, 4)
        
        # Compute rotation loss via cross entropy
        loss_rot = tf.nn.softmax_cross_entropy_with_logits(labels=labels_rot, logits=logits_rot)
        loss_rot = tf.reduce_mean(loss_rot)
    else:
        loss_rot = 0.0

    return loss_rot


@tf.function
def compute_kl_loss(model, u_augment, labels_u, w_kl=0):
    """
    Compute Kullback-Leibler Loss baed on unlabeled augmented input images.

    Args:
        model:      tf.keras model
        u_augment:  tensor, augmented unlabeled batch of images of shape [batch, height, width, channels] 
        labels_u:   tensor, labels of unlabeled batch of images u of shape [batch, num_classes]
        w_kl:       float, if > 0.0 KL loss will be computed else 0 will be returned

    Returns:
        float, KL loss based u_augment and its labels.
    """
    if w_kl > 0:
        # Compute model output of u_augment
        logits_u_aug = model(u_augment, training=True)[0]

        # Compute KL loss
        loss_kl = tf.nn.softmax_cross_entropy_with_logits(labels=labels_u, logits=logits_u_aug)
        loss_kl = tf.reduce_mean(loss_kl)
    else:
        loss_kl = 0.0

    return loss_kl


def remixmatch(model, x, y, u, T, K, beta, height, width, w_kl=0.5):
    """
    Applies remixmatch algorithm on inputs x, y and u returns mixmatched tensors
    X_prime and U_prime and a float holding the Kullback-Leibler loss.

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
        w_kl:       float, if > 0 w_kl loss will be computed else set to 0

    Returns:
        Two tensors, one holding the transformed input images consisting of a
        transformation of the mixed labeled and unlabeled data, the other one
        holds its corresponding aggregated labels.
        Additionally float, holding KL Loss will be returned
    """
    # print(x.shape) # x -> xt_in
    # print(y.shape) # y -> l_in
    # print(u.shape) # u -> y_in
    batch_size = x.shape[0]

    x_aug = weak_augment(x, height, width)
    u_aug = [None for _ in range(K)]

    for k in range(K):
        u_aug[k] = weak_augment(u, height, width)
    u_aug_labels = guess_labels(u_aug, model, K)
    u_aug_labels = sharpen(u_aug_labels, tf.constant(T))
    
    kl_loss = compute_kl_loss(model, u_aug[0], u_aug_labels, w_kl)
    
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

    return X_prime, U_prime, kl_loss

@tf.function
def ssl_loss_remixmatch(labels_x, logits_x, labels_u, logits_u, use_xeu=True):
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
        use_xeu:    Boolean, if True CE Loss of unlabeled images will be computed else MSE

    Returns:
        Two floating point numbers, the first holding the labeled CE loss, the 
        second holding either the unlabeled CE loss or the MSE of the unlabeled
        images.
    """
    loss_x = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
    loss_x = tf.reduce_mean(loss_x)
    
    if use_xeu:
        loss_xeu = tf.nn.softmax_cross_entropy_with_logits(labels=labels_u, logits=logits_u)
    else:
        loss_xeu = tf.square(labels_u - tf.nn.softmax(logits_u))
    loss_xeu = tf.reduce_mean(loss_xeu)
    
    return loss_x, loss_xeu
