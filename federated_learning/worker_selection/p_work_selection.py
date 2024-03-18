from .selection_strategy import SelectionStrategy


def jaccard_similarity(list1, list2):
    """
    Jaccard distance [28] to measure the distance between two attribute sets.
    """

    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))
def n_largest_index(lst, n):
    sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
    return sorted_indices[:n]


class PaperSelectionStrategy(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """
    def select_round_workers(self, workers, poisoned_workers, kwargs):
        # Database Reputation
        jaccard_sim = [jaccard_similarity(list(range(6)), worker.get_attributes_name()) for worker in workers]
        n_largest_jaccard_sim = n_largest_index(jaccard_sim, kwargs["NUM_WORKERS_PER_ROUND"])
        print(n_largest_jaccard_sim)

        # Training Reputation

        return n_largest_jaccard_sim
