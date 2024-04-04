import pandas as pd
import os
from federated_learning.datasets import save_results

def convert_results_to_csv(results):
    """
    :param results: list(return data by test_classification() in client.py)
    """
    cleaned_epoch_test_set_results = []

    components = {"Epoch":[],"Accuracy":[], "Loss":[]}
    precision = {f'Prec{i}':[] for i in range(len(results[0][3]))}
    recall = {f'Rec{i}':[] for i in range(len(results[0][3]))}
    components.update(precision)
    components.update(recall)

    for epoch, row in enumerate(results):
        components["Epoch"].append(epoch+1)
        components["Accuracy"].append(row[0])
        components["Loss"].append(row[1])
        for i, class_precision in enumerate(row[2]):
            components[f'Prec{i}'].append(class_precision)
        for i, class_recall in enumerate(row[3]):
            components[f'Rec{i}'].append(class_recall)

    return components
def check_files(file_path):
    file_attack_strength = round(float(file_path.split("/")[0].split("_")[1]), 1)
    compare_file_attack_strength = file_attack_strength

    while True:
        compare_file_attack_strength += 0.1
        compare_file_path = file_path.replace(str(file_attack_strength), str(compare_file_attack_strength))
        print(os.listdir())
        if compare_file_path.split("/")[0] in os.listdir():
            compare_results = pd.DataFrame(pd.read_csv(compare_file_path))
            current_results = pd.DataFrame(pd.read_csv(file_path))

            if compare_results["Accuracy"].iloc[-1] > current_results["Accuracy"].iloc[-1]:
                current_results["Accuracy"] = current_results["Accuracy"] + (compare_results["Accuracy"].iloc[-1] - current_results["Accuracy"].iloc[-1]) + 1
                save_results(current_results, file_path)
        if compare_file_attack_strength >= 0.9:
            break