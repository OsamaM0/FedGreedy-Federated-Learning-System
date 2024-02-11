def distribute_batches_non_iid(train_data_loader, num_workers):
    """
    Gives each worker batches of training data of a specific class.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]
    label_to_worker = {}

    for batch_idx, (data, target) in enumerate(train_data_loader):
        # Get the label of the first sample in the batch
        label = target[0].item()

        # Assign the batch to a worker based on the label
        if label not in label_to_worker:
            label_to_worker[label] = len(label_to_worker) % num_workers

        worker_idx = label_to_worker[label]
        distributed_dataset[worker_idx].append((data, target))

    return distributed_dataset