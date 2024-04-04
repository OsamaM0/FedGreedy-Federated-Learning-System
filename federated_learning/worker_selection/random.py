from .selection_strategy import SelectionStrategy
import random

from ..utils import identify_random_elements

class RandomSelectionStrategy(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """

    def select_round_workers(self, workers, poisoned_workers, kwargs):

        workers_idx = [worker.get_client_index() for worker in workers]

        # Get Poisoned and Non-Poisoned Workers
        poisoned_workers_idx = identify_random_elements(len(workers), int((kwargs["NUM_POISONED_WORKERS"] / kwargs["NUM_WORKERS_PER_ROUND"]) * 50))

        # Set the random seed based on current_epoch_number
        random.seed(kwargs["current_epoch_number"])

        not_poisoned_worker = random.sample(
            [workers_id for workers_id in workers_idx if workers_id not in poisoned_workers_idx],
            kwargs["NUM_WORKERS_PER_ROUND"] - kwargs["NUM_POISONED_WORKERS"])

        poisoned_worker = random.sample(
            [workers_id for workers_id in poisoned_workers_idx if workers_id in poisoned_workers_idx],
            kwargs["NUM_POISONED_WORKERS"])

        return not_poisoned_worker + poisoned_worker
