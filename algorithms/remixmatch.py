from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

from . import mixup


def random_rotate(x):
    b4 = x.shape[0] // 4
    l = np.zeros(b4, np.int32)
    l = tf.constant(np.concatenate([l, l + 1, l + 2, l + 3], axis=0))
    return tf.concat([x[:b4], tf.image.rot90(x[b4:2 * b4], k=1), tf.image.rot90(x[2 * b4:3 * b4], k=2), tf.image.rot90(x[3 * b4:], k=3)], axis=0), l


def compute_rot_loss(x, model, w_rot=0.5):
    if w_rot > 0:
        y_rot, labels_rot = random_rotate(x)
        logits_rot = tf.keras.layers.Dense(4, kernel_initializer="glorot_normal")(model(y_rot, training=True)[1])
        labels_rot = tf.one_hot(labels_rot, 4)
        loss_rot = tf.nn.softmax_cross_entropy_with_logits(labels=labels_rot, logits=logits_rot)
        loss_rot = tf.reduce_mean(loss_rot)
    else:
        loss_rot = 0

    return loss_rot


def compute_kl_loss(model, u_augment_labels, u_augment, w_kl):
    if w_kl > 0:
        u_aug_out = model(u_augment, training=True)[0]
        loss_kl = tf.nn.softmax_cross_entropy_with_logits(labels=u_augment_labels, logits=u_aug_out)
        loss_kl = tf.reduce_mean(loss_kl)
    else:
        loss_kl = 0

    return loss_kl


def remixmatch(model, x, y, u, T, K, beta, height, width, w_kl=0.5):
    # print(x.shape) # x -> xt_in
    # print(y.shape) # y -> l_in
    # print(u.shape) # u -> y_in
    batch_size = x.shape[0]

    x_aug = augment(x, height, width)
    u_aug = [None for _ in range(K)]

    for k in range(K):
        u_aug[k] = augment(u, height, width)
    u_aug_labels = guess_labels(u_aug, model, K)
    u_aug_labels = sharpen(u_aug_labels, tf.constant(T))
    
    kl_loss = compute_kl_loss(model, u_aug_labels, u_aug[0], w_kl)
    
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
    loss_x = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
    loss_x = tf.reduce_mean(loss_x)
    
    if use_xeu:
        loss_xeu = tf.nn.softmax_cross_entropy_with_logits(labels=labels_u, logits=logits_u)
    else:
        loss_xeu = tf.square(labels_u - tf.nn.softmax(logits_u))
    loss_xeu = tf.reduce_mean(loss_xeu)
    
    return loss_x, loss_xeu
