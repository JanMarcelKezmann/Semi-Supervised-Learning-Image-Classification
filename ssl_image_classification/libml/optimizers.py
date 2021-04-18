import tensorflow as tf


def get_optimizer(opt_name, lr=0.01, momentum=0.9, beta1=0.9, beta2=0.999, rho=0.95, eps=1e-07):
    """
    Defines and returns tf.keras Optimizer.

    Args:
        opt_name:       string, name of optimizer, choose one of ('AdaDelta', 'AdaGrad',
                        'Adam', 'Adamax', 'Nadam', 'RMSProp', 'SGD')
        lr:             float or lr schedule, learning rate (schedule)
        momentum:       float, hyperparameter >= 0 that accelerates gradient descent in the relevant direction and dampens oscillations.
        beta1:          float, exponential decay rate for the 1st moment estimates
        beta2:          float, exponential decay rate for the 2nd moment estimates
        rho:            float, decay rate used in 'Adadelta' and 'RMSProp'
        eps:            float, small constant epsilon used to better conditioning the grad update

    Returns:
        tf.keras.optimizers.Optimizer


    """
    
    if opt_name.lower() == "adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate=lr, rho=rho, epsilon=eps, name='Adadelta')
    elif opt_name.lower() == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=lr, initial_accumulator_value=0.1, epsilon=eps, name='Adagrad')
    elif opt_name.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta1, epsilon=eps, amsgrad=False, name='Adam')
    elif opt_name.lower() == "adamax":
        return tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=eps, name='Adamax')
    elif opt_name.lower() == "nadam":
        returntf.keras.optimizers.Nadam(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=eps, name='Nadam')
    elif opt_name.lower() == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, momentum=momentum, epsilon=eps, centered=False, name='RMSprop')
    elif opt_name.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True, name='SGD')
    else:
        raise ValueError(f"{opt_name} must be one of 'AdaDelta', 'AdaGrad', 'Adam', 'Adamax', 'Nadam', 'RMSProp' or 'SGD'.")