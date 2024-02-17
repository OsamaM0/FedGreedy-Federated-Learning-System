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
