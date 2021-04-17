import tensorflow as tf

from ..libml.data_augmentations import weak_augment, medium_augment, strong_augment


def ict(x, u, height, width):
    """
    Applies medium augmentations on inputs x and u returns augmented tensors.

    Args:
        x:          tensor, labeled batch of images of shape [batch, height, width, channels]
        u:          tensor, unlabeled batch of images of shape [batch, height, widht, channels]
        height:     int, height of images
        width:      int, width of images

    Returns:
        Augmented labeled tensor and two augmented unlabeled tensors.
    """
    x_augment = medium_augment(x, height, width)
    u_teacher = weak_augment(u, height, width)
    u_student = medium_augment(u, height, width)

    return x_augment, u_teacher, u_student


def ssl_loss_ict(labels_x, logits_x, labels_teacher, logits_student):
    """
    Computes two cross entropy losses based on the labeled and unlabeled data.
    loss_x is referring to the labeled CE loss and loss_u to the unlabeled CE loss.

    Args:
        labels_x:       tensor, contains labels corresponding to logits_x of shape [batch, num_classes]
        logits_x:       tensor, contains the logits of an batch of images of shape [batch, num_classes]
        labels_teacher: tensor, labels of teacher model of shape [batch, num_classes]
        labels_student: tensor, logits of student model of shape [batch, num_classes]

    Returns:
        Two floating point numbers, the first representing the labeled CE loss
        and the second holding the MSE loss values.
    """
    x_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
    x_loss = tf.reduce_mean(x_loss)

    ict_loss = tf.reduce_mean((labels_teacher - tf.nn.softmax(logits_student)) ** 2, -1)
    ict_loss = tf.reduce_mean(pm_loss)
    
    return x_loss, loss_ict