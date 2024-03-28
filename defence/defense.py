
import os
from collections import Counter

from loguru import logger
from federated_learning.arguments import Arguments
from defence.dimensionality_reduction import calculate_pca_of_gradients
from federated_learning.model.parameters import get_layer_parameters
from federated_learning.model.parameters import calculate_parameter_gradients
from federated_learning.utils import get_model_files_for_epoch
from federated_learning.utils import get_model_files_for_suffix
from defence import apply_standard_scaler
from federated_learning.utils import get_worker_num_from_model_file_name
from client import Client
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


from collections import Counter
from federated_learning.arguments import Arguments
from client import Client

EXP_INFO_PATH = "logs/1823.log"
# The layer of the NNs that you want to investigate.
#   If you are using the provided Fashion MNIST CNN, this should be "fc.weight"
#   If you are using the provided Cifar 10 CNN, this should be "fc2.weight"
#   If you are using the provided HAPT DNN, this should be "softmax.weight"
LAYER_NAME = "fc_f.weight"
CLASS_NUM = 4
POISONED_WORKER_IDS = []
SAVE_NAME = "defense_results.jpg"
SAVE_SIZE = (18, 14)


def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))
        clients.append(client)
    return clients


def identify_outliers(data, k=2, n_repeats=11):
    kmeans = KMeans(n_clusters=k)
    cluster_labels_list = [kmeans.fit(data).labels_ for _ in range(n_repeats)]
    most_common_labels = Counter(tuple(labels) for labels in cluster_labels_list).most_common(1)[0][0]
    silhouette_avg = silhouette_score(data, most_common_labels)

    if silhouette_avg < 0.2:
        k = 1
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(data)
        return cluster_labels

    cluster_points = {cluster: [] for cluster in range(k)}
    for labels in cluster_labels_list:
        for i, cluster in enumerate(labels):
            cluster_points[cluster].append(data[i])

    smallest_cluster = min(cluster_points, key=lambda x: len(cluster_points[x]))
    indices_smallest_cluster = [i for i, cluster_label in enumerate(most_common_labels) if
                                cluster_label == smallest_cluster]
    return indices_smallest_cluster

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_gradients_3d(gradients, poisoned_worker_ids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (worker_id, gradient) in gradients:
        color = "blue" if worker_id in poisoned_worker_ids else "orange"
        marker = "x" if worker_id in poisoned_worker_ids else "o"
        ax.scatter(gradient[0], gradient[1], gradient[2], color=color, marker=marker, s=100 if marker == "x" else 180)
        ax.text(gradient[0], gradient[1], gradient[2], str(worker_id), color='black')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

def get_poisoned_worker(epoch, path):
    args = Arguments(logger)
    MODELS_PATH = path
    model_files = sorted(os.listdir(MODELS_PATH))
    start_model_files = get_model_files_for_epoch(model_files, epoch)
    start_model_file = get_model_files_for_suffix(start_model_files, args.get_cr_save_start_suffix())[0]
    start_model_file = os.path.join(MODELS_PATH, start_model_file)
    start_model = load_models(args, [start_model_file])[0]
    start_model_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

    end_model_files = get_model_files_for_epoch(model_files, epoch)
    end_model_files = get_model_files_for_suffix(end_model_files, args.get_cr_save_end_suffix())

    param_diff = []
    worker_ids = []
    for end_model_file in end_model_files:
        worker_id = get_worker_num_from_model_file_name(end_model_file)
        end_model_file = os.path.join(MODELS_PATH, end_model_file)
        end_model = load_models(args, [end_model_file])[0]
        end_model_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])
        gradient = calculate_parameter_gradients(logger, start_model_layer_param, end_model_layer_param)
        gradient = gradient.flatten()
        param_diff.append(gradient)
        worker_ids.append(worker_id)

    scaled_param_diff = StandardScaler().fit_transform(param_diff)
    dim_reduced_gradients = PCA(n_components=4).fit_transform(scaled_param_diff)

    poisoned_worker_idx = [worker_ids[i] for i in identify_outliers(dim_reduced_gradients, k=2)]
    logger.info(f"Poisoned worker ids\n\t\t\t: {poisoned_worker_idx}")
    # plot_gradients_3d(zip(worker_ids, dim_reduced_gradients), poisoned_worker_idx)
    return poisoned_worker_idx