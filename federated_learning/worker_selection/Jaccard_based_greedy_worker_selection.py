import copy

from .selection_strategy import SelectionStrategy
import  random
from federated_learning.arguments import Arguments


# TOTAL_LBL = set(range(6))  # assuming labels are from 0 to 5    => HAPT

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

class JaccardGreedySelectionStrategy(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """
    random.seed(42)
    def filter_and_sort_clients(self, workers, TOTAL_LBL ):
        valid_clients = {worker_idx: set(labels.get_attributes_name()) for worker_idx, labels in enumerate(workers) if
                         set(labels.get_attributes_name()).issubset(TOTAL_LBL)}
        sorted_clients = sorted(valid_clients,
                                key=lambda x: (jaccard_similarity(valid_clients[x], TOTAL_LBL), -len(valid_clients[x])),
                                reverse=True)
        print("Sorted Clients: ", sorted_clients)
        return sorted_clients, valid_clients
    def select_round_workers(self, workers, poisoned_workers, kwargs):
        TOTAL_LBL = set(range(kwargs["LABELS_NUM"]))
        sorted_clients, valid_clients = self.filter_and_sort_clients(workers, TOTAL_LBL)
        print(valid_clients)
        # Greedy algorithm to select clients covering all labels
        selected_clients = list()
        selected_labels = set()
        round_lbl_selected = 0
        while True:
            for client in sorted_clients:
                # If you get all labels, stop the loop
                if len(selected_labels) == len(TOTAL_LBL) or ( round_lbl_selected > 1 and len(selected_clients) >= int(kwargs["NUM_WORKERS_PER_ROUND"])):
                    # If Selected Labels Get All Labels Start Collecting them from the beginning
                    if len(selected_clients) > int(kwargs["NUM_WORKERS_PER_ROUND"]) and  round_lbl_selected >= 1:
                        selected_clients = selected_clients[ : kwargs["NUM_WORKERS_PER_ROUND"]]
                        print("Final Selected Worker: ", [workers[i].get_client_index() for i in  selected_clients])
                        return [workers[i].get_client_index() for i in  selected_clients]
                    elif len(selected_clients) == int(kwargs["NUM_WORKERS_PER_ROUND"]):
                        print("Final Selected Worker: ", [workers[i].get_client_index() for i in  selected_clients])
                        return [workers[i].get_client_index() for i in  selected_clients]
                    elif len(selected_clients) < int(kwargs["NUM_WORKERS_PER_ROUND"]):
                        # Set the selected labels to empty to start collecting them from the beginning
                        selected_labels = set()
                        round_lbl_selected += 1
                        print("Round Selected Labels: ", round_lbl_selected)
                        break
                    else:
                        random.shuffle(workers)
                        return self.select_round_workers(workers, poisoned_workers, kwargs)


                # Take the client if it is not launch attack more than 2 times and it has labels that are not already selected
                new_labels = valid_clients[client] - selected_labels
                if new_labels:
                    if poisoned_workers.count(client) <= 2:
                        if client not in selected_clients:
                            selected_clients.append(client)
                            selected_labels.update(new_labels)
                            print("Selected labels: ", selected_labels)
                            print("Selected clients: ", [workers[i].get_client_index() for i in  selected_clients])
                            # Re-sort the clients and valid clients by default mode
                            sorted_clients, valid_clients = self.filter_and_sort_clients(workers, TOTAL_LBL)
                            break
                        else:
                            continue
                    else:
                        # Re-sort the clients and valid clients to new mode to find better clients but for this label
                        random.seed(min(list(new_labels)))
                        temp_worker = copy.deepcopy(workers)
                        random.shuffle(temp_worker)
                        sorted_clients, valid_clients = self.filter_and_sort_clients(workers, TOTAL_LBL)
                        break
            else:
                random.shuffle(workers)
                return self.select_round_workers(workers, poisoned_workers, kwargs)
                # raise ValueError("Maximum number of clients reached without covering all labels")

