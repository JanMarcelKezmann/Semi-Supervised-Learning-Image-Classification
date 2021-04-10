# Imports
import tensorflow as tf


@tf.function
def mixup(x1, x2, y1, y2, beta, alg):
    if alg.lower() == "mixmatch":
        beta = tf.maximum(beta, 1 - beta)
        x = beta * x1 + (1 - beta) * x2
        y = beta * y1 + (1 - beta) * y2
    elif alg.lower() == "mixup":
        beta = tf.maximum(beta, 1 - beta)
        x = beta * x1 + (1 - beta) * x2
        y = beta[:, :, 0, 0] * y1 + (1 - beta[:, :, 0, 0]) * y2
        
    return x, y


@tf.function
def ssl_loss(labels_x, logits_x, labels_u, logits_u):
    loss_x = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
    loss_x = tf.reduce_mean(loss_x)
    loss_u = tf.nn.softmax_cross_entropy_with_logits(labels=labels_u, logits=logits_u)
    loss_u = tf.reduce_mean(loss_u)

    return loss_x, loss_u