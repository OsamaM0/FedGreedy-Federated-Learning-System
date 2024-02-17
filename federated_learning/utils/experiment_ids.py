import os
import pathlib
def generate_experiment_ids(start_idx, num_exp):
    """
    Generate the filenames for all experiment IDs.

    :param start_idx: start index for experiments
    :type start_idx: int
    :param num_exp: number of experiments to run
    :type num_exp: int
    """
    log_files = []
    results_files = []
    models_folders = []
    reputation_worker_selections_files = []
    data_worker_selections_files = []

    for i in range(num_exp):
        # idx = str(start_idx + i)
        idx = start_idx

        if not os.path.exists("stats"):
            pathlib.Path("stats").mkdir()
        if not os.path.exists("logs"):
            pathlib.Path("logs").mkdir()

        log_files.append("logs/" + idx + ".log")
        results_files.append("stats/" +idx + "_results.csv")
        models_folders.append(idx + "_models")
        reputation_worker_selections_files.append("stats/" + idx + "reputation_workers_selected.csv")
        data_worker_selections_files.append("stats/" + idx + "data_workers_selected.csv")

    return log_files, results_files, models_folders, reputation_worker_selections_files, data_worker_selections_files
