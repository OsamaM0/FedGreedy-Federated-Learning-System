from loguru import logger


def apply_class_label_replacement(X, Y, replacement_method, poison_strength=1):
    """
    Replace class labels using the replacement method

    :param X: data features
    :type X: numpy.Array()
    :param Y: data labels
    :type Y: numpy.Array()
    :param replacement_method: Method to update targets
    :type replacement_method: method
    """
    logger.debug("Applying class label replacement")
    return (X, replacement_method(Y, set(Y), poison_strength))
