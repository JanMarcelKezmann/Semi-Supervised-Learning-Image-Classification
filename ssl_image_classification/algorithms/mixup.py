# Imports
import tensorflow as tf


@tf.function
def mixup(x1, x2, y1, y2, beta, alg):
    """
    Applies mixup algorithm to input images and its corresponding labels and
    returns them.

    Args:
        x1:     tensor, 1st batch of images
        x2:     tensor, 2nd batch of images
        y1:     tensor, 1st batch of labels for corresponding images x1
        y2:     tensor, 2nd batch of labels for corresponding images x2
        beta:   tensor, random vector distributed accoriding to Beta distribution
        alg:    string, Either 'mixup', 'mixmatch' or 'vat' depending on used ssl algorithm

    Returns:
        Two tensors, the first containing the new batch of images and
        the second containing its correpsonding labels.        
    """
    beta = tf.maximum(beta, 1 - beta)
    if alg.lower() == "mixmatch":
        x = beta * x1 + (1 - beta) * x2
        y = beta * y1 + (1 - beta) * y2
    elif alg.lower() in ["mixup", "vat"]:
        x = beta * x1 + (1 - beta) * x2
        y = beta[:, :, 0, 0] * y1 + (1 - beta[:, :, 0, 0]) * y2
        
    return x, y


@tf.function
def ssl_loss_mixup(labels_x, logits_x, labels_u, logits_u):
    """
    Computes two cross entropy losses based on the labeled and unlabeled data.
    loss_x is referring to the labeled CE loss and loss_u to the unlabeled CE loss.

    Args:
        labels_x:   tensor, contains labels corresponding to logits_x of shape [batch, num_classes]
        logits_x:   tensor, contains the logits of an batch of images of shape [batch, num_classes]
        labels_u:   tensor, contains labels corresponding to logits_u of shape [batch, num_classes]
        logits_u:   tensor, contains the logits of an batch of images of shape [batch, num_classes]
        
    Returns:
        Two floating point numbers, representing the CE loss values.
    """
    loss_x = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
    loss_x = tf.reduce_mean(loss_x)
    loss_u = tf.nn.softmax_cross_entropy_with_logits(labels=labels_u, logits=logits_u)
    loss_u = tf.reduce_mean(loss_u)

    return loss_x, loss_u
