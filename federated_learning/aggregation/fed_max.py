def max_nn_parameters(parameters, acc, selected_workers):
    """
    Max passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    max_acc = acc.index(max(acc))
    print("Selected Worker: ",selected_workers[max_acc], " - Accuracy: ", acc[max_acc])
    return parameters[max_acc]
