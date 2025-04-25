from tabulate import tabulate

def eval_predictions_all_labels(df, include_not_originally_downloaded=True, only_accuracy=False):
    total = 0
    correct = 0
    false_predictions = 0

    # What was the target label and what did the model predict (first hierarchy is target label, second is model label)
    type_predictions = {
        'unsubstantiate': {
            'total': 0,
            'unsubstantiate': 0,
            'partially substantiate': 0,
            'fully substantiate': 0,
            'invalid label': 0,
        },
        'partially substantiate': {
            'total': 0,
            'unsubstantiate': 0,
            'partially substantiate': 0,
            'fully substantiate': 0,
            'invalid label': 0,
        },
        'fully substantiate': {
            'total': 0,
            'unsubstantiate': 0,
            'partially substantiate': 0,
            'fully substantiate': 0,
            'invalid label': 0,
        }
    }

    for index, row in df.iterrows():
        if row['Reference Article Downloaded'] == 'Yes':
            if include_not_originally_downloaded or row['Reference Article PDF Available'] == 'Yes':
                total += 1
                target_label = row['Label']
                type_predictions[target_label]['total'] += 1
                model_label = row['Model Classification Label']

                if model_label not in ['unsubstantiate', 'partially substantiate', 'fully substantiate']:
                    false_predictions += 1
                    type_predictions[target_label]['invalid label'] += 1
                    print(f"Row {index} Model Classification Label is not a valid label: {model_label}")
                    continue

                type_predictions[target_label][model_label] += 1

                if target_label == model_label:
                    correct += 1
                else:
                    false_predictions += 1

    if only_accuracy:
        return {
            'accuracy': round(correct / total, 3)
        }
    else:
        return {
            'accuracy': round(correct / total, 3),
            'total': total,
            'correct': correct,
            'false_predictions': false_predictions,
            'type_predictions': type_predictions
        }

def replace_substantiate_label(label):
    if label in ['partially substantiate', 'fully substantiate']:
        label = 'substantiate'
    return label

def eval_predictions_two_labels(df, include_not_originally_downloaded=True, only_accuracy=False):
    total = 0
    correct = 0
    false_predictions = 0
    false_positives = 0
    false_negatives = 0

    for index, row in df.iterrows():
        if row['Reference Article Downloaded'] == 'Yes':
            if include_not_originally_downloaded or row['Reference Article PDF Available'] == 'Yes':
                total += 1
                target_label = replace_substantiate_label(row['Label'])
                model_label = replace_substantiate_label(row['Model Classification Label'])

                if target_label == model_label:
                    correct += 1
                else:
                    false_predictions += 1
                    if model_label not in ['unsubstantiate', 'substantiate']:
                        print(f"Row {index} Model Classification Label is not a valid label: {model_label}")
                    elif target_label == 'unsubstantiate' and model_label == 'substantiate':
                        false_positives += 1
                    elif target_label == 'substantiate' and model_label == 'unsubstantiate':
                        false_negatives += 1
    
    if only_accuracy:
        return {
            'accuracy': round(correct / total, 3)
        }
    else:
        return {
            'accuracy': round(correct / total, 3),
            'total': total,
            'correct': correct,
            'false_predictions': false_predictions,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
def calc_label_accuracies(results, exclude_not_available=False):
    # Initialize dictionary
    label_accuracies = {}
    for model in results:
        label_accuracies[model] = {
            'unsubstantiate': None,
            'partially substantiate': None,
            'fully substantiate': None,
            'overall': None,
        }
    
    for model, model_results in results.items():
        category_name = 'all_labels' + ('_exclude_not_available' if exclude_not_available else '')
        type_predictions = model_results[category_name]['type_predictions']
        for label in type_predictions:
            if type_predictions[label]['total'] > 0:
                label_accuracies[model][label] = round(type_predictions[label][label] / type_predictions[label]['total'], 3)
            else:
                label_accuracies[model][label] = 0
        label_accuracies[model]['overall'] = round(model_results[category_name]['accuracy'], 3)
    
    return label_accuracies

def print_table_label_accuracies(label_accuracies):
    # Prepare data for the table
    table_data = [
        [model, 
        f"{accuracies['unsubstantiate'] * 100:.1f}", 
        f"{accuracies['partially substantiate'] * 100:.1f}", 
        f"{accuracies['fully substantiate'] * 100:.1f}", 
        f"{accuracies['overall'] * 100:.1f}"]
        for model, accuracies in label_accuracies.items()
    ]

    # Define headers
    headers = ['Model', 'Un', 'Partially', 'Fully', 'Overall']

    # Display the table
    print(tabulate(table_data, headers=headers, tablefmt='pretty'))