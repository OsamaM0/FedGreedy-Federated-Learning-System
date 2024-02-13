from loguru import logger

import plots
from federated_learning.arguments import Arguments
from federated_learning.datasets import generate_data_loaders_from_distributed_dataset
from federated_learning.aggregation import average_nn_parameters
from attack import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.datasets import save_results
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from federated_learning.utils import load_pickle_file
from client import Client

def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)

    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        clients[client_idx].train(epoch)

    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    new_nn_params = average_nn_parameters(parameters)

    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)

    return clients[0].test(), random_workers

def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients

def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected = train_subset_of_clients(epoch, args, clients, poisoned_workers)

        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)

    return convert_results_to_csv(epoch_test_set_results), worker_selection

def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    # 1. Get the User Arguments
    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.set_data_distribution("non_iid")
    args.log()

    # 2. Load the Train and Test Datasets
    data_distribution = args.get_data_distribution()
    train_data_loader_path = args.get_train_data_loader_pickle_path().split("/")
    train_data_loader_path.insert(-1, data_distribution)
    print(train_data_loader_path)
    args.set_train_data_loader_pickle_path("/".join(train_data_loader_path))

    test_data_loader_path = args.get_test_data_loader_pickle_path().split("/")
    test_data_loader_path.insert(-1, data_distribution)
    args.set_test_data_loader_pickle_path("/".join(test_data_loader_path))


    distributed_train_dataset = load_pickle_file(args.get_train_data_loader_pickle_path())
    test_data_loader = load_pickle_file(args.get_test_data_loader_pickle_path())

    plots.plot_data_distribution(distributed_train_dataset)

    # 4. Random Choose Clients and Poison their Train Datasets
    poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method)

    # 5. Generate Dataloader for Both poisend and trained datasets
    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    # 6. Assign Train & Test Dataset to Clients
    clients = create_clients(args, train_data_loaders, test_data_loader)

    # 7. Start Federated Learning
    results, worker_selection = run_machine_learning(clients, args, poisoned_workers)

    # 8. Save Results
    save_results(results, results_files[0])
    save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)
