import tensorflow as tf

def get_model(name="ResNet50", weights=None, height=None, width=None,
    classes=None, include_top=True, pooling=None, alpha=1.0,
    classifier_activation=None, depth_multiplier=1.0, dropout=0.001):
    """
    Returns the tf.keras.applications model of choise with weight, height, width and further configurations.

    Args:
        name:                   string, name of the model
        weights:                One of None (random initialization), 'imagenet' (pre-training on ImageNet)
        height:                 int, height of model inputs
        width:                  int, width of model inputs
        classes:                int, number of classes
        include_top:            Boolean, Whether to include the fully-connected layer at the top of the network.
        pooling:                Optional pooling mode for feature extraction when include_top is False. Options are one of [None, "avg", "max"]
        classifier_activation:  A str or callable. The activation function to use on the "top" layer. Ignored unless include_top=True.
                                    Set classifier_activation=None to return the logits of the "top" layer.
        alpha:                  float, between 0 and 1, controls the width of the network. (MobilieNet and MobileNetV2 only)
        depth_multiplier:       float, depth multiplier for depthwise convolution. (MobileNet only)
        dropout:                float, dropout rate (MobileNet only)

    Returns:
        tf.keras.applications Model
    """

    if not isinstance(height, int) or not isinstance(width, int):
       raise TypeError("'height' and 'width' need to be of type 'int'")

    
    if name.lower() == "densenet121":
        if height <= 31 or width <= 31:
          raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.DenseNet121(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "densenet169":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.DenseNet169(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "densenet201":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.DenseNet201(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "efficientnetb0":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB0(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "efficientnetb1":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB1(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "efficientnetb2":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB2(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "efficientnetb3":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB3(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "efficientnetb4":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB4(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "efficientnetb5":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB5(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "efficientnetb6":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB6(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "efficientnetb7":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB7(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "inceptionresnetv2":
        if height <= 74 or width <= 74:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 75.")
        base_model = tf.keras.applications.InceptionResNetV2(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "inceptionv3":
        if height <= 74 or width <= 74:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 75.")
        base_model = tf.keras.applications.InceptionV3(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "mobilenet":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.MobileNet(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, alpha=alpha, depth_multiplier=depth_multiplier, dropout=dropout, classifier_activation=classifier_activation)
    elif name.lower() == "mobilenetv2":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.MobileNetV2(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, alpha=alpha, classifier_activation=classifier_activation)
    elif name.lower() == "nasnetlarge":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.NASNetLarge(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "nasnetmobile":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.NASNetMobile(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "resnet50":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet50(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "resnet50v2":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet50V2(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "resnet101":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet101(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "resnet101v2":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet101V2(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "resnet152":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet152(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "resnet152v2":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet152V2(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "vgg16":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.VGG16(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "vgg19":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.VGG19(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    elif name.lower() == "xception":
        if height <= 70 or width <= 70:
            raise ValueError("Parameters 'height' and width' should not be smaller than 71.")
        base_model = tf.keras.applications.Xception(include_top=include_top, classes=classes, weights=weights, input_shape=[height, width, 3], pooling=pooling, classifier_activation=classifier_activation)
    else:
        raise ValueError("'name' should be one of 'densenet121', 'densenet169', 'densenet201', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', \
                'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', \
                'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2', 'nasnetlarge', 'nasnetmobile', \
                'resnet50', 'resnet50v2', 'resnet101', 'resnet101v2', 'resnet152', 'resnet152v2', 'vgg16', 'vgg19' or 'xception'.")
    
    model = tf.keras.Model(base_model.inputs, [base_model.output, base_model.layers[-2].output])
    return model