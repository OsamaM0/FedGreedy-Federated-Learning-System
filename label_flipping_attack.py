from loguru import logger

from attack.label_filpping import replace_0_with_2
from attack.label_filpping import replace_5_with_3
from attack.label_filpping import replace_1_with_9
from attack.label_filpping import replace_4_with_6
from attack.label_filpping import replace_1_with_3
from attack.label_filpping import replace_6_with_0
from attack.label_filpping import global_replace
from federated_learning.worker_selection import RandomSelectionStrategy, PaperSelectionStrategy, JaccardGreedySelectionStrategy
from server import run_exp

if __name__ == '__main__':

    START_EXP_IDX = 3000
    NUM_EXP = 1
    ALGORITHM = "fedprox"
    REPLACEMENT_METHOD = global_replace
    NUM_POISONED_WORKERS = 0
    # for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
    for attacker_num in [0]:
        for strength_num in [0]:
            for algorithm in ["fed_greedy", "fed_prox", "fed_avg", "fed_max"]:
                for data_distribution in ["non_iid", "iid"]:
                    experiment_id =[f"{attacker_num}_{strength_num}", f"{algorithm}_{data_distribution}"]
                    logger.info(f"Running experiment with Attack Number = {attacker_num} \t- Attack Strength = {strength_num} \t- Algorithm = {algorithm} \t- Data Distribution = {data_distribution}")
                    KWARGS = {
                        "NUM_WORKERS_PER_ROUND": 5,
                        "NUM_POISONED_WORKERS": attacker_num,
                        "STRENGTH_OF_POISON": strength_num,
                        "REPLACEMENT_METHOD": global_replace
                    }
                    NUM_POISONED_WORKERS = int(attacker_num * 50)
                    ALGORITHM = algorithm
                    if ALGORITHM == "fed_greedy":
                        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, JaccardGreedySelectionStrategy(),data_distribution, experiment_id)
                    elif ALGORITHM == "fed_avg":
                        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, RandomSelectionStrategy(),data_distribution, experiment_id)
                    elif ALGORITHM == "fed_prox":
                        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, JaccardGreedySelectionStrategy(),data_distribution, experiment_id)
                    elif ALGORITHM == "fed_max":
                        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, PaperSelectionStrategy(),data_distribution, experiment_id)
