from .selection_strategy import SelectionStrategy
import random

class RandomSelectionStrategy(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """

    def select_round_workers(self, workers, poisoned_workers, kwargs):
        # Set the random seed
        random.seed(kwargs["current_epoch_number"])
        workers_id = [worker.get_client_index() for worker in workers]
        return random.sample(workers_id, kwargs["NUM_WORKERS_PER_ROUND"])
