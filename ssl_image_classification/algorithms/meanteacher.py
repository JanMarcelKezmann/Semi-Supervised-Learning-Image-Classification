import tensorflow as tf

from ..libml.data_augmentations import weak_augment, medium_augment, strong_augment


def mean_teacher(x, u, height, width, seed=[29598, 29598]):
    """
    Applies medium augmentations on inputs x, y and u returns tensors
    and returns augmented tensors.

    Args:
        x:          tensor, labeled batch of images [batch, height, width, channels]
        u:          tensor, unlabeled batch of images [batch, height, widht, channels]
        height:     int, height of images
        width:      int, width of images
        seed:       int, seed for replicating augmentations

    Returns:
        Augmented input tensors.
    """
    x_aug = medium_augment(x, height, width)
    u_t = medium_augment(u, height, width, seed=seed)
    u_s = medium_augment(u, height, width, seed=seed)

    return x_aug, u_t, u_s


@tf.function
def ssl_loss_mean_teacher(labels_x, logits_x, logits_teacher, logits_student):
    """
    Computes two cross entropy losses based on the labeled and unlabeled data.
    loss_x is referring to the labeled CE loss and loss_u to the unlabeled CE loss.

    Args:
        labels_x:       tensor, contains labels corresponding to logits_x of shape [batch, num_classes]
        logits_x:       tensor, contains the logits of an batch of images of shape [batch, num_classes]
        logits_teacher: tensor, logits of teacher model of shape [batch, num_classes]
        labels_student: tensor, logits of student model of shape [batch, num_classes]

    Returns:
        Two floating point numbers, the first representing the labeled CE loss
        and the second holding the MSE loss values.
    """
    x_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
    x_loss = tf.reduce_mean(x_loss)

    loss_mt = tf.reduce_mean((tf.nn.softmax(logits_teacher) - tf.nn.softmax(logits_student)) ** 2, -1)
    loss_mt = tf.reduce_mean(loss_mt)

    return x_loss, loss_mt