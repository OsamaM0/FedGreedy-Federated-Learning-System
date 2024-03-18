from loguru import logger

from attack.label_filpping import replace_0_with_2
from attack.label_filpping import replace_5_with_3
from attack.label_filpping import replace_1_with_9
from attack.label_filpping import replace_4_with_6
from attack.label_filpping import replace_1_with_3
from attack.label_filpping import replace_6_with_0
from attack.label_filpping import global_replace_hapt
from federated_learning.worker_selection import RandomSelectionStrategy, PaperSelectionStrategy, JaccardGreedySelectionStrategy
# from server import run_exp
from server import run_exp


if __name__ == '__main__':

    REPLACEMENT_METHOD = global_replace_hapt
    NUM_WORKER_PER_ROUND = 10


    for attacker_num in range(5,0,-1):
        for strength_num in range(9,0,-1):
            strength_num = strength_num * 0.1
            for algorithm in ["fed_greedy", "fed_avg", "fed_prox",  "fed_max"]:
                for data_distribution in ["iid","non_iid"]:
                    experiment_id =[f"{attacker_num}_{strength_num}", f"{algorithm}_{data_distribution}"]
                    logger.info(f"Running experiment {experiment_id}")
                    KWARGS = {
                        "NUM_WORKERS_PER_ROUND": NUM_WORKER_PER_ROUND,
                        "NUM_POISONED_WORKERS": attacker_num,
                        "STRENGTH_OF_POISON": strength_num,
                        "REPLACEMENT_METHOD": global_replace_hapt
                    }
                    NUM_POISONED_WORKERS =  int((attacker_num / NUM_WORKER_PER_ROUND ) * 50)
                    ALGORITHM = algorithm
                    if ALGORITHM == "fed_greedy":
                        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, JaccardGreedySelectionStrategy(),data_distribution, experiment_id)
                    elif ALGORITHM == "fed_avg":
                        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, RandomSelectionStrategy(),data_distribution, experiment_id)
                    elif ALGORITHM == "fed_prox":
                        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, RandomSelectionStrategy(),data_distribution, experiment_id)
                    elif ALGORITHM == "fed_max":
                        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, PaperSelectionStrategy(),data_distribution, experiment_id)