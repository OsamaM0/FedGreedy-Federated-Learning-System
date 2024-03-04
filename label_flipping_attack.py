from attack.label_filpping import replace_0_with_2
from attack.label_filpping import replace_5_with_3
from attack.label_filpping import replace_1_with_9
from attack.label_filpping import replace_4_with_6
from attack.label_filpping import replace_1_with_3
from attack.label_filpping import replace_6_with_0
from federated_learning.worker_selection import RandomSelectionStrategy, PaperSelectionStrategy, JaccardGreedySelectionStrategy
from server import run_exp

if __name__ == '__main__':

    START_EXP_IDX = 3000
    NUM_EXP = 1
    NUM_POISONED_WORKERS = 0
    ALGORITHM = "fedprox"
    REPLACEMENT_METHOD = replace_1_with_3
    KWARGS = {
        "NUM_WORKERS_PER_ROUND": 10
    }

    # for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
    for data_distribution in ["non_iid", "iid"]:
        for algorithm in ["fed_greedy", "fed_avg", "fed_prox", "fed_max"]:
            experiment_id = f"{algorithm}_{data_distribution}"

            ALGORITHM = algorithm
            if ALGORITHM == "fed_greedy":
                run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, JaccardGreedySelectionStrategy(),data_distribution, experiment_id)
            elif ALGORITHM == "fed_avg":
                run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, RandomSelectionStrategy(),data_distribution, experiment_id)
            elif ALGORITHM == "fed_prox":
                run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, RandomSelectionStrategy(),data_distribution, experiment_id)
            elif ALGORITHM == "fed_max":
                run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, ALGORITHM, PaperSelectionStrategy(),data_distribution, experiment_id)
