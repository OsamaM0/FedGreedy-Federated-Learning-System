import math

from loguru import logger

import plots
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
def train_subset_of_clients(args, round, clients, poisoned_workers, clients_repitition):
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

    """ CLIENT SELECTION STRATEGY"""
    # Check first if there is any poisoned data you want to replace

    selected_worker = args.get_round_worker_selection_strategy().select_round_workers(
        clients,
        # poisoned_workers,
        kwargs)

    """ TRAINING THE CLIENTS """
    clients_struggle = [] # list of client struggle
    straggler_epochs = max(int(epochs * (args.get_struggling_epochs_percentage())), 1) # detect number of straggler epochs >10 -> 5(S)
    num_straggler = args.get_struggling_epochs_percentage() * len(selected_worker)     # detect number of straggler clietns > 6 -> 3(S)
    # 1, 2, 3, 4, 5
    # 1
    for i, client_idx in enumerate(selected_worker) :
        args.get_logger().info("Training Round #{} on client #{}", str(round), str(clients[client_idx].get_client_index()))

        # itrate over not straggler clients
        if (i <= math.floor(len(selected_worker) - int(num_straggler))):
            clients[client_idx].train(round, epochs, algorithm )  # Train
        # itrate over straggler clients
        elif algorithm == "fed_prox":
            print("Worker #{} is a straggler".format(client_idx))
            clients[client_idx].train(round, straggler_epochs, algorithm )
            clients_struggle.append(client_idx)

        # Evaluation
        if client_idx in clients_repitition.keys():
            clients_repitition[client_idx].append(clients[client_idx].test()[0])
        else:
            clients_repitition[client_idx] = [None] * (round - 1) + [clients[client_idx].test()[0]]

    for client_idx in clients_repitition.keys():
        if len(clients_repitition[client_idx]) < round:
            clients_repitition[client_idx].append(None)

    """ MODEL AGGREGATION STRATEGY"""
    args.get_logger().info("Aggregate client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in selected_worker]

    if algorithm == "fed_max":
        clients_acc = [clients[client_idx].test()[0] for client_idx in selected_worker]
        new_nn_params = max_nn_parameters(parameters, clients_acc, selected_worker)
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

def run_machine_learning(clients, args):
    """
    Complete machine learning over a series of clients.
    """
    cr_test_set_results = []
    clients_repitition = {}
    clients_poisoned = []
    clients_data = {}

    for round in range(1, args.get_num_cr() + 1):
        results, clients_repitition, clients_struggle = train_subset_of_clients(args, round ,clients, clients_poisoned, clients_repitition)

        prop_poisoned_workers = get_poisoned_worker(round, args.get_save_model_folder_path() )

        # Determine which workers were selected
        for client, values in clients_repitition.items():
            if client not in clients_data:
                clients_data[client] = [None] * (round - 1)
            if values[-1] is not None:
                # Determine if a worker has been poisoned
                if len(values) >= 2 and values[-2] is not None and values[-1] is not None and values[-2] - values[-1] > 5 and client not in clients_struggle:
                    if client in prop_poisoned_workers and args.get_algorithm() == "fed_prox":
                        clients_poisoned.append(client)
                        clients_data[client].append("poisoned")
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
    log_files, results_files, models_folders, reputation_selections_files, data_worker_file = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    # 1. Get the User Arguments
    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.set_data_distribution(data_distribution)
    args.set_algorithm(algorithm)
    args.log()

    # 2. Load the Train and Test Datasets
    data_distribution = args.get_data_distribution()
    train_data_loader_path = args.get_train_data_loader_pickle_path().split("/")
    train_data_loader_path.insert(-1, data_distribution)
    args.set_train_data_loader_pickle_path("/".join(train_data_loader_path))

    test_data_loader_path = args.get_test_data_loader_pickle_path().split("/")
    test_data_loader_path.insert(-1, data_distribution)
    args.set_test_data_loader_pickle_path("/".join(test_data_loader_path))


    distributed_train_dataset = load_pickle_file(args.get_train_data_loader_pickle_path())
    test_data_loader = load_pickle_file(args.get_test_data_loader_pickle_path())

    # plots.plot_data_distribution(distributed_train_dataset)

    # 4. Random Choose Clients and Poison their Train Datasets
    poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method)

    # 5. Generate Dataloader for Both poisend and trained datasets
    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    # 6. Assign Train & Test Dataset to Clients
    clients = create_clients(args, train_data_loaders, test_data_loader)

    # 7. Start Federated Learning
    results,  worker_data, worker_reputation = run_machine_learning(clients, args)

    # 8. Save Results
    save_results(results, results_files[0])
    save_results(worker_reputation, reputation_selections_files[0])
    print(worker_data)
    save_results(worker_data, data_worker_file[0])

    logger.remove(handler)
