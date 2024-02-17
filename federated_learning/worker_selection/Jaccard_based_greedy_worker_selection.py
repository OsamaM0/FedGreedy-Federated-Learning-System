from .selection_strategy import SelectionStrategy
import  random

TOTAL_LBL = set(range(10))  # assuming labels are from 0 to 9 => MNIST, CIFAR10, FashionMNIST
# TOTAL_LBL = set(range(6))  # assuming labels are from 0 to 5    => HAPT

# Function to calculate Jaccard distance between client's labels and total labels
def jaccard_distance(client_labels):
    return len(client_labels.intersection(TOTAL_LBL)) / len(client_labels.union(TOTAL_LBL))

class JaccardGreedySelectionStrategy(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """
    def select_round_workers(self, workers, poisoned_workers, kwargs):
        # Filter and sort valid clients based on Jaccard distance
        valid_clients = {worker_idx: set(labels.get_attributes_name()) for worker_idx, labels in enumerate(workers) if
                         set(labels.get_attributes_name()).issubset(TOTAL_LBL)}
        sorted_clients = sorted(valid_clients, key=lambda x: jaccard_distance(valid_clients[x]), reverse=True)

        # Greedy algorithm to select clients covering all labels
        selected_clients = set()
        selected_labels = set()

        for client in sorted_clients:
            print(poisoned_workers)
            if poisoned_workers.count(client) <= 1:
                new_labels = valid_clients[client] - selected_labels
                if new_labels:
                    selected_clients.add(client)
                    selected_labels.update(new_labels)
                    print("Selected labels: ", selected_labels)
                    print("Selected clients: ", selected_clients)
                if len(selected_labels) == len(TOTAL_LBL):
                    if len(selected_clients) == kwargs["NUM_WORKERS_PER_ROUND"]:
                        break
                    else:
                        selected_clients.add(client)
        else:
            random.shuffle(workers)
            self.select_round_workers(workers, poisoned_workers, kwargs)
            # raise ValueError("Maximum number of clients reached without covering all labels")

        return list(selected_clients)
