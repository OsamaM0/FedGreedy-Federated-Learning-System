import numpy as np

def distribute_batches_non_iid(train_data_loader, num_workers, num_classes=6):
    """
    Distributes batches of training data with Non-IID Label Distribution among workers.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    :param num_classes: number of classes
    :type num_classes: int
    :return: list of distributed datasets for each worker
    :rtype: list
    """
    distributed_dataset = [[] for _ in range(num_workers)]

    # Generate random label subsets for each worker without replacement
    label_subsets = [np.random.choice(num_classes, size=np.random.randint(2,3), replace=False) for _ in range(num_workers)]

    # Track the used labels to avoid duplicates

    for batch_idx, (data, target) in enumerate(train_data_loader):
        used_labels = set()
        # Shuffle the order of workers
        worker_order = np.random.permutation(num_workers)

        for worker_idx in worker_order:
            worker_labels = label_subsets[worker_idx]

            # If any label in the worker's subset has already been used, skip this worker
            if any(label in used_labels for label in worker_labels):
                continue

            # Filter data and labels based on worker's labels
            filtered_indices = list(np.where(np.isin(target.numpy(), worker_labels))[0])
            filtered_data = data[filtered_indices]
            filtered_target = target[filtered_indices]
            if len(filtered_target) != 0:
                distributed_dataset[worker_idx].extend(list(zip(filtered_data, filtered_target)))

            # Update used labels set
            used_labels.update(worker_labels)

            # Remove assigned samples from the batch
            data = np.delete(data, filtered_indices, axis=0)
            target = np.delete(target, filtered_indices, axis=0)

            # If all samples are assigned, break the loop
            if len(data) == 0:
                break

        distributed_batch_dataset = [[] for _ in range(num_workers)]
        # Make Batches of samples
        for worker_id, worker_data in enumerate(distributed_dataset):
            worker_batch_sample_data = []
            worker_batch_sample_target = []
            for sample_idx, (data, target) in enumerate(worker_data):
                worker_batch_sample_data.append(data)
                worker_batch_sample_target.append(target)
                if len(worker_batch_sample_data) == 10:
                    distributed_batch_dataset[worker_id].append((worker_batch_sample_data,worker_batch_sample_target) )
                    worker_batch_sample_data = []
                    worker_batch_sample_target = []
            # else:
            #     distributed_batch_dataset[worker_id].append(worker_batch_sample)

    return distributed_batch_dataset