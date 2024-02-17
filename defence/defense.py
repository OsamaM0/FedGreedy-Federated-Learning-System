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

# Paths you need to put in.

EXP_INFO_PATH = "logs/1823.log"

# The layer of the NNs that you want to investigate.
#   If you are using the provided Fashion MNIST CNN, this should be "fc.weight"
#   If you are using the provided Cifar 10 CNN, this should be "fc2.weight"
#   If you are using the provided HAPT DNN, this should be "softmax.weight"
LAYER_NAME = "fc.weight"

# The source class.
CLASS_NUM = 4

# The IDs for the poisoned workers. This needs to be manually filled out.
# You can find this information at the beginning of an experiment's log file.
POISONED_WORKER_IDS =[]

# The resulting graph is saved to a file
SAVE_NAME = "defense_results.jpg"
SAVE_SIZE = (18, 14)


def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))

        clients.append(client)

    return clients


from sklearn.metrics import silhouette_score

def identify_outliers(data, k=2, n_repeats=3):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k)

    # Repeat fitting multiple times
    cluster_labels_list = []
    for _ in range(n_repeats):
        cluster_labels = kmeans.fit(data).labels_
        cluster_labels_list.append(cluster_labels)

    # Find the most common clustering result
    from collections import Counter
    most_common_labels = Counter(tuple(labels) for labels in cluster_labels_list).most_common(1)[0][0]

    # Evaluate silhouette score for the most common clustering
    silhouette_avg = silhouette_score(data, most_common_labels)

    # If silhouette score indicates low separability, consider single cluster
    if silhouette_avg < 0.2:
        k = 1
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(data)
        return cluster_labels

    # Identify points belonging to each cluster
    cluster_points = {cluster: [] for cluster in range(k)}
    for labels in cluster_labels_list:
        for i, cluster in enumerate(labels):
            cluster_points[cluster].append(data[i])

    # Find the smallest cluster
    smallest_cluster = min(cluster_points, key=lambda x: len(cluster_points[x]))

    # Return indices of data points in the smallest cluster
    indices_smallest_cluster = [i for i, cluster_label in enumerate(most_common_labels) if
                                cluster_label == smallest_cluster]

    return indices_smallest_cluster


def plot_gradients_2d(gradients):
    fig, ax = plt.subplots()

    for (worker_id, gradient) in gradients:
        if worker_id in POISONED_WORKER_IDS:
            ax.scatter(gradient[0], gradient[1], color="blue", marker="x", s=1000, linewidth=5)
        else:
            ax.scatter(gradient[0], gradient[1], color="orange", s=180)
        ax.annotate(worker_id, (gradient[0], gradient[1]), textcoords="offset points", xytext=(0,10), ha='center')

    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.grid(False)
    plt.margins(2,2)
    plt.show()
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)



def get_poisoned_worker(epoch, path):

    args = Arguments(logger)
    args.log()
    MODELS_PATH = path
    model_files = sorted(os.listdir(MODELS_PATH))
    logger.debug("Number of models: {}", str(len(model_files)))

    param_diff = []
    worker_ids = []

    start_model_files = get_model_files_for_epoch(model_files, epoch)

    start_model_file = get_model_files_for_suffix(start_model_files, args.get_cr_save_start_suffix())[0]
    start_model_file = os.path.join(MODELS_PATH, start_model_file)
    start_model = load_models(args, [start_model_file])[0]

    start_model_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

    end_model_files = get_model_files_for_epoch(model_files, epoch)
    end_model_files = get_model_files_for_suffix(end_model_files, args.get_cr_save_end_suffix())
    for end_model_file in end_model_files:
        worker_id = get_worker_num_from_model_file_name(end_model_file)
        end_model_file = os.path.join(MODELS_PATH, end_model_file)
        end_model = load_models(args, [end_model_file])[0]

        end_model_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

        gradient = calculate_parameter_gradients(logger, start_model_layer_param, end_model_layer_param)
        gradient = gradient.flatten()

        param_diff.append(gradient)
        worker_ids.append(worker_id)


    # logger.info("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))

    # logger.info("Prescaled gradients: {}".format(str(param_diff)))
    scaled_param_diff = apply_standard_scaler(param_diff)
    # logger.info("Postscaled gradients: {}".format(str(scaled_param_diff)))
    dim_reduced_gradients = calculate_pca_of_gradients(logger, scaled_param_diff, 2)
    # logger.info("PCA reduced gradients: {}".format(str(dim_reduced_gradients)))
    #
    # logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))

    #======================================================
    # Function to identify points not belonging to any cluster
    poisoned_worker_idx = [ worker_ids[i] for i in identify_outliers(dim_reduced_gradients, k=2)]
    if len(poisoned_worker_idx) == 0: poisoned_worker_idx = ["-"]
    print("POISONED WORKERS: ", poisoned_worker_idx)
    # plot_gradients_2d(zip(worker_ids, dim_reduced_gradients))
    return poisoned_worker_idx
    #========================= =============================
