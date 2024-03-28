import copy
import math

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from federated_learning.model.nets import FashionMNISTCNN, HAPTDNN
import plots
from federated_learning.aggregation.fed_greedy_max import avg_max_nn_parameters
from federated_learning.arguments import Arguments
from federated_learning.datasets import generate_data_loaders_from_distributed_dataset
from federated_learning.aggregation import average_nn_parameters, max_nn_parameters
from attack import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.datasets import save_results
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from federated_learning.utils import load_pickle_file
from client import Client
from defence import get_poisoned_worker
import random


selected_worker = []
clients = []
def train_subset_of_clients(args, round, poisoned_workers, clients_repitition, struggle_workers):
    """
    Train a subset of clients per round.

    :param round: round
    :type round: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = round
    epochs = args.get_epoch()
    algorithm = args.get_algorithm()
    cr = args.get_num_cr()

    """ CLIENT SELECTION STRATEGY"""
    # Check first if there is any poisoned data you want to replace
    global selected_worker
    global clients

    """ SELECT THE CLIENTS FOR THE ROUND """
    if algorithm in ["fed_avg", "fed_prox"]:
        """ SELECT THE CLIENTS FOR THE ROUND """
        selected_worker = args.get_round_worker_selection_strategy().select_round_workers(
            clients,
            poisoned_workers,
            kwargs)
        logger.info("Clients: {}", [clients[client_idx].get_client_index() for client_idx in selected_worker])

    """ TRAINING THE CLIENTS """
    clients_struggle = [] # list of client struggle
    straggler_epochs = max(int(epochs * (1 - args.get_struggling_epochs_percentage())), 1) # detect number of straggler epochs >10 -> 5(S)
    num_straggler = np.ceil(args.get_struggling_workers_percentage() * len(selected_worker))     # detect number of straggler clietns > 6 -> 3(S)

    for i, client_idx in enumerate(selected_worker) :
        client_id = clients[client_idx].get_client_index()
        client = clients[client_idx]

        if client_id not in struggle_workers:
            args.get_logger().info("Training Round #{} on client #{}", str(round), str(client_id))

            # Iterate over not straggler clients
            if (i <= math.floor(len(selected_worker) - int(num_straggler))):
                # Check if the client is poisoned and the algorithm is fed_greedy then train the client
                if client_id in poisoned_workers:
                    client.set_mu( client.get_mu()*2 )
                    logger.info("Worker #{} is a poisoned worker during training", client_id)
                    logger.info("Mu #{} is: ", client.mu)
                    client.train(round, epochs, algorithm)  # Train
                # elif client.get_mu() > args.get_mu()*2:
                elif client.get_mu() > args.get_mu()*2:
                    client.train(round, epochs, algorithm)  # Train
                else:
                    # Change the algorithm name to fed_avg if the algorithm is fed_greedy and not poisoned or straggler
                    client.train(round, epochs, "fed_avg" )  # Train

            # Iterate over straggler clients
            elif algorithm in ["fed_greedy", "fed_prox"]:
                print("Worker #{} is a straggler during training".format(client_id))
                client.train(round, straggler_epochs, algorithm )
                clients_struggle.append(client_id)

            # Evaluation
            if client_id in clients_repitition.keys():
                clients_repitition[client_id].append(client.test()[0])
            else:
                clients_repitition[client_id] = [None] * (round - 1) + [client.test()[0]]
        else:
            print("Worker #{} is a straggler".format(client_id))

    """ CALCULATE THE REPETITION OF THE CLIENTS"""
    for client_idx in clients_repitition.keys():
        if len(clients_repitition[client_idx]) < round:
            clients_repitition[client_idx].append(None)

    """ MODEL AGGREGATION STRATEGY"""
    args.get_logger().info("Aggregate client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in selected_worker]
    clients_acc = [clients[client_idx].test()[0] for client_idx in selected_worker]

    if algorithm == "fed_max":
            new_nn_params = max_nn_parameters(parameters, clients_acc, selected_worker)

    elif algorithm == "fed_greedy":
        if (round / cr) < 1 :
            parameters = [clients[client_idx].get_nn_parameters() for client_idx in selected_worker if clients[client_idx].test()[0] > sum(clients_acc) / len(clients_acc) - 4]
            new_nn_params = average_nn_parameters(parameters)
        else:
            print("avg_max_nn_parameters")
            clients_acc = [clients[client_idx].test()[3] for client_idx in selected_worker]
            new_nn_params = avg_max_nn_parameters(parameters, clients_acc, selected_worker)

    else:
        new_nn_params = average_nn_parameters(parameters)


    """MODEL WEIGHT CLIENT UPDATE"""
    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)

    return clients[0].test(log=True), clients_repitition, clients_struggle

def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients

def run_machine_learning(args, struggle_workers):
    """
    Complete machine learning over a series of clients.
    """
    cr_test_set_results = []
    clients_repitition = {}
    clients_poisoned = []
    clients_data = {}


    for c_round in range(1, args.get_num_cr() + 1):

        results, clients_repitition, clients_struggle = train_subset_of_clients(args, copy.deepcopy(c_round) , clients_poisoned, clients_repitition, struggle_workers)

        # Track the poisoned workers after each epoch
        if args.get_algorithm() == "fed_greedy" and c_round > 1:
                prop_poisoned_workers = get_poisoned_worker(c_round, args.get_save_model_folder_path())
        else:
            prop_poisoned_workers = []

        clients_poisoned = []
        # Determine which workers were selected
        for client, values in clients_repitition.items():
            if client not in clients_data:
                clients_data[client] = [None] * (c_round - 1)
            if values[-1] is not None:
                # Determine if a worker has been poisoned
                if len(values) >= 2 and values[-2] is not None  and values[-2] - values[-1] > 3 and client not in clients_struggle:
                    if client in prop_poisoned_workers:
                        clients_poisoned.append(client)
                        clients_data[client].append("poisoned")
                        logger.info("Worker #{} Added to the poisoned workers", client)
                    else:
                        clients_data[client].append("normal")
                # Determine if a worker is a straggler
                elif client in clients_struggle:
                    clients_data[client].append("struggler")
                # Determine if a worker is normal
                else:
                    clients_data[client].append("normal")
            else:
                clients_data[client].append(None)

        # Evaluate the model Results
        cr_test_set_results.append(results)
    return convert_results_to_csv(cr_test_set_results), clients_data, clients_repitition









def run_exp(replacement_method, num_poisoned_workers, KWARGS, algorithm, client_selection_strategy, data_distribution, idx):
    print(idx)
    log_files, results_files, models_folders, reputation_selections_files, data_worker_file = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)
    # Get the User Arguments
    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.set_data_distribution(data_distribution)
    args.set_algorithm(algorithm)
    global selected_worker
    selected_worker = []
    global  clients
    clients = []
    args.log()
    attack_strength = KWARGS["STRENGTH_OF_POISON"]
    struggle_workers = [40, 7, 1, 47, 17, 15, 14, 8, 6, 43, 34, 5, 37, 27, 2, 13, 32, 38, 35, 12, 45, 41, 44, 26, 28]
    lbl_classes = KWARGS["LABELS_NUM"]
    if lbl_classes == 6:
        args.net = HAPTDNN
        args.train_data_loader_pickle_path = "data_loaders/hapt/train_data_loader.pickle"
        args.test_data_loader_pickle_path = "data_loaders/hapt/test_data_loader.pickle"

    else:
        args.net = FashionMNISTCNN
        args.train_data_loader_pickle_path = "data_loaders/fashion-mnist/train_data_loader.pickle"
        args.test_data_loader_pickle_path = "data_loaders/fashion-mnist/test_data_loader.pickle"
    #===================================================================================================================
    #========================================= Start the Federated Learning ============================================
    #===================================================================================================================

    #--------------------------------------------------- Client Selection Strategy -------------------------------------
    # 1.1. Load the Train and Test Datasets
    data_distribution = args.get_data_distribution()

    train_data_loader_path = args.get_train_data_loader_pickle_path().split("/")
    train_data_loader_path.insert(-1, data_distribution)
    args.set_train_data_loader_pickle_path("/".join(train_data_loader_path))

    test_data_loader_path = args.get_test_data_loader_pickle_path().split("/")
    test_data_loader_path.insert(-1, data_distribution)
    args.set_test_data_loader_pickle_path("/".join(test_data_loader_path))


    distributed_train_dataset = load_pickle_file(args.get_train_data_loader_pickle_path())
    test_data_loader = load_pickle_file(args.get_test_data_loader_pickle_path())
    #
    # plots.plot_data_distribution(distributed_train_dataset, lbl_classes)

    # 1.2. Poison Data for clients and create the data loaders
    if args.get_algorithm() in ["fed_greedy", "fed_max"]:
        # Distribute the data to the clients for clients selection
        data_distribution = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())
        clients = create_clients(args, data_distribution, test_data_loader)
        # Remove the struggling workers
        logger.info("Struggle Workers: {}", struggle_workers)
        clients_temp = [client for client in clients if client.get_client_index() not in struggle_workers]
        logger.info("Clients: {}", [client.get_client_index() for client in clients_temp] )
        selected_worker = args.get_round_worker_selection_strategy().select_round_workers(
            clients_temp,
            [],
            KWARGS)
        # After selecting the clients, poison the data of the selected clients
        poisoned_workers_idx = selected_worker[len(selected_worker) - KWARGS["NUM_POISONED_WORKERS"]:]
        logger.info("Poisoned Workers: {}", poisoned_workers_idx)
        distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), lbl_classes, poisoned_workers_idx, replacement_method, attack_strength)
        data_distribution = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())
        clients = create_clients(args, data_distribution, test_data_loader)

    elif args.get_algorithm() in ["fed_avg", "fed_prox"]:
        poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
        logger.info("Poisoned Workers: {}", poisoned_workers)
        distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(),lbl_classes, poisoned_workers, replacement_method, attack_strength)
        data_distribution = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())
        clients = create_clients(args, data_distribution, test_data_loader)

    # 7. Start Federated Learning
    results,  worker_data, worker_reputation = run_machine_learning(args, struggle_workers)

    # 8. Save Results
    save_results(results, results_files[0])
    save_results(worker_reputation, reputation_selections_files[0])
    save_results(worker_data, data_worker_file[0])

    logger.remove(handler)
