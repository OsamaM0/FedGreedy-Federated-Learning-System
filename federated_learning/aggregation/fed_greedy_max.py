from .fed_avg import average_nn_parameters
import numpy as np

def avg_max_nn_parameters(parameters, acc, selected_workers):
    """
    Max passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    acc = np.random.permutation(acc)
    selected_workers_idx = [max(enumerate(col), key=lambda x: x[1])[0] for col in zip(*acc)]
    print("Selected Worker for Aggregation: ",[selected_workers[i] for i in selected_workers_idx], " - Accuracy: ", [acc[i] for i in selected_workers_idx])
    return average_nn_parameters([parameters[i] for i in selected_workers_idx])