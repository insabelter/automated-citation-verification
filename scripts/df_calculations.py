from tabulate import tabulate
import pandas as pd

def eval_predictions(df, include_relabelled_partially=False, include_not_originally_downloaded=True, only_accuracy=False):
    G = 0
    P = 0
    N = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    invalid_labels = {
        'Unsubstantiated': [],
        'Substantiated': []
    }

    for index, row in df.iterrows():
        if row['Reference Article Downloaded'] == 'Yes':
            if include_not_originally_downloaded or row['Reference Article PDF Available'] == 'Yes':
                if include_relabelled_partially or row['Previously Partially Substantiated'] != 'x':
                    G += 1
                    target_label = row['Label']
                    model_label = row['Model Classification Label']

                    assert target_label in ['Unsubstantiated', 'Substantiated'], f"Row {index} Label is not a valid label: {target_label}"

                    invalid_label = False
                    if model_label not in ['Unsubstantiated', 'Substantiated']:
                        invalid_labels[target_label].append(model_label)
                        print(f"Row {index} Model Classification Label is not a valid label: {model_label}")
                        invalid_label = True
                
                    if target_label == 'Substantiated':
                        P += 1
                        if model_label == 'Substantiated':
                            TP += 1
                        elif model_label == 'Unsubstantiated':
                            FN += 1
                        elif invalid_label:
                            FN += 1
                    elif target_label == 'Unsubstantiated':
                        N += 1
                        if model_label == 'Substantiated':
                            FP += 1
                        elif model_label == 'Unsubstantiated':
                            TN += 1
                        elif invalid_label:
                            FP += 1   

    assert G == P + N, f"Total G ({G}) does not equal P ({P}) + N ({N})"
    assert TP + FN == P, f"TP ({TP}) + FN ({FN}) does not equal P ({P})"
    assert TN + FP == N, f"TN ({TN}) + FP ({FP}) does not equal N ({N})"

    accuracy = (TP + TN) / G
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / P if P > 0 else 0.0
    specificity = TN / N if N > 0 else 0.0
    f1_score = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    if only_accuracy:
        return {
            'accuracy': round(accuracy, 3)
        }
    else:
        return {
            'G (Total)': G,
            'P (Substantiated)': P,
            'N (Unsubstantiated)': N,
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'accuracy': round(accuracy, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'specificity': round(specificity, 3),
            'f1_score': round(f1_score, 3),
            'invalid_labels': invalid_labels
        }
    
# def count_preds_for_label(type_predictions_dict, label):
#     """
#     Count the number of predictions for a specific label in a results dictionary.
#     """
#     label_count = 0
#     for _, count in type_predictions_dict.items():
#         label_count += count[label]
#     return label_count

# def print_table_label_accuracies(label_accuracies, string_given=False, two_labels=False):
#     if two_labels:
#         if string_given:
#             table_data = [
#                 [model, 
#                 f"{accuracies['unsubstantiate']}", 
#                 f"{accuracies['fully substantiate']}", 
#                 f"{accuracies['overall']}"]
#                 for model, accuracies in label_accuracies.items()
#             ]
#         else:
#             table_data = [
#                 [model, 
#                 f"{accuracies['unsubstantiate'] * 100:.1f}", 
#                 f"{accuracies['fully substantiate'] * 100:.1f}", 
#                 f"{accuracies['overall'] * 100:.1f}"]
#                 for model, accuracies in label_accuracies.items()
#             ]
#     else:
#         if string_given:
#             table_data = [
#                 [model, 
#                 f"{accuracies['unsubstantiate']}", 
#                 f"{accuracies['partially substantiate']}", 
#                 f"{accuracies['fully substantiate']}", 
#                 f"{accuracies['overall']}"]
#                 for model, accuracies in label_accuracies.items()
#             ]
#         else:
#             table_data = [
#                 [model, 
#                 f"{accuracies['unsubstantiate'] * 100:.1f}", 
#                 f"{accuracies['partially substantiate'] * 100:.1f}", 
#                 f"{accuracies['fully substantiate'] * 100:.1f}", 
#                 f"{accuracies['overall'] * 100:.1f}"]
#                 for model, accuracies in label_accuracies.items()
#             ]

#     # Define headers
#     if two_labels:
#         headers = ['Model', 'Un', 'Fully', 'Overall']
#     else:
#         headers = ['Model', 'Un', 'Partially', 'Fully', 'Overall']

#     # Display the table
#     print(tabulate(table_data, headers=headers, tablefmt='pretty'))

# def calc_preds_per_error_type(df):
#     """
#     Calculates the number total, correct and false class predictions for the unsubstantiated rows per error type in the DataFrame.
#     """
#     error_types = list(df['Error Type'][df['Error Type'].notna()].unique())
#     error_types.sort()

#     preds_per_error_type = {
#         error_type: {
#             "total": 0,
#             "correct_class": 0,
#             "false_class": 0
#         } for error_type in error_types
#     }

#     for _, row in df[df['Label'] == 'unsubstantiate'].iterrows():
#         assert pd.notna(row['Error Type']), f"Error Type is NaN for row: {row}"
#         error_type = row['Error Type']
#         preds_per_error_type[error_type]['total'] += 1
#         if row['Model Classification Label'] == row['Label']:
#             preds_per_error_type[error_type]['correct_class'] += 1
#         else:
#             preds_per_error_type[error_type]['false_class'] += 1

#     return preds_per_error_type

def display_model_results_table(model_results_dict, use_pandas=True):
    """
    Display model evaluation results as a formatted table.
    
    Parameters:
    model_results_dict (dict): Dictionary where keys are model names and values are result dictionaries
                              containing accuracy, precision, recall, specificity, and f1_score
    
    Returns:
    pd.DataFrame: DataFrame with the results formatted as a table
    """
    # Extract the metrics we want to display
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    
    # Create a list to store the table data
    table_data = []
    
    for model_name, results in model_results_dict.items():
        row = [model_name]  # Start with model name
        for metric in metrics:
            if metric in results:
                # Format as decimal with 3 decimal places
                row.append(f"{results[metric]:.3f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Create column headers
    headers = ['Model'] + [metric.capitalize() for metric in metrics]
    
    # Display the table using tabulate
    if not use_pandas:
        print("Model Performance Comparison")
        print("=" * 70)
        print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center'))
    
    # Also create and return a DataFrame for further analysis
    df_data = []
    for model_name, results in model_results_dict.items():
        row = {'Model': model_name}
        for metric in metrics:
            if metric in results:
                row[metric.capitalize()] = results[metric]
            else:
                row[metric.capitalize()] = None
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.set_index('Model', inplace=True)
    
    if use_pandas:
        display(df)

def get_preds_results(results):
    return {
        "Unsubstantiated": { # negative class
            "preds": results['TN'] + results['FN'],
            "correct_preds": results['TN'],
            "correct_total": results['N (Unsubstantiated)'],
        },
        "Substantiated": { # positive class
            "preds": results['TP'] + results['FP'],
            "correct_preds": results['TP'],
            "correct_total": results['P (Substantiated)'],
        },
    }