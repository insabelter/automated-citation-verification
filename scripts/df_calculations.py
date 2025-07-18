from tabulate import tabulate
import pandas as pd
from IPython.display import display

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
    assert G != 0, f"Total G ({G}) is 0, cannot calculate metrics"

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

def eval_predictions_per_attribute_value(df, attribute, include_relabelled_partially):
    results = {}
    results['Total'] = eval_predictions(df, include_relabelled_partially=include_relabelled_partially)
    for value in df[attribute].unique():
        df_value = df[df[attribute] == value]
        results_value = eval_predictions(df_value, include_relabelled_partially=include_relabelled_partially)
        results[value] = results_value
    return results

def display_attribute_results_table(results_dict, attribute, use_pandas=True):
    """
    Display a table of model results by attribute values showing only accuracy.
    
    Parameters:
    results_dict: Dictionary with structure {model_name: {attribute_value: {results...}}}
    use_pandas: If True, use pandas DataFrame display; if False, use tabulate
    
    Returns:
    pandas.DataFrame: The results table
    """
    
    # Get all unique attribute values from the first model
    first_model = list(results_dict.keys())[0]
    attribute_values = list(results_dict[first_model].keys())
    
    # Create data for the table
    table_data = []
    for model_name, model_results in results_dict.items():
        row = [model_name]
        accuracies = []
        for attr_value in attribute_values:
            if attr_value in model_results and 'accuracy' in model_results[attr_value]:
                accuracy = model_results[attr_value]['accuracy']
                row.append(f"{accuracy * 100:.1f}%")
                accuracies.append(accuracy)
            else:
                row.append("N/A")
                accuracies.append(None)
        
        table_data.append(row)
    
    # Create DataFrame
    columns = ['Model', 'Total'] + [f'"{value_name}"' for value_name in attribute_values if value_name != 'Total']
    df = pd.DataFrame(table_data, columns=columns)
    df.set_index('Model', inplace=True)
    
    print(f"Results for attribute '{attribute}':")
    # Display based on preference
    if use_pandas:
        display(df)
    else:
        print(tabulate(df, headers=df.columns, tablefmt='grid', stralign='center'))

def display_attribute_differences_to_total_table(results_dict, attribute, use_pandas=True):
    """
    Display a table showing the difference between each attribute value accuracy and the total accuracy.
    
    Parameters:
    results_dict: Dictionary with structure {model_name: {attribute_value: {results...}}}
    attribute: The attribute name being analyzed
    use_pandas: If True, use pandas DataFrame display; if False, use tabulate
    
    Returns:
    pandas.DataFrame: The results table with differences to total
    """
    
    # Get all unique attribute values from the first model (excluding 'Total')
    first_model = list(results_dict.keys())[0]
    attribute_values = [val for val in results_dict[first_model].keys() if val != 'Total']
    
    # Create data for the table
    table_data = []
    for model_name, model_results in results_dict.items():
        row = [model_name]
        
        # Get total accuracy for this model
        total_accuracy = None
        if 'Total' in model_results and 'accuracy' in model_results['Total']:
            total_accuracy = model_results['Total']['accuracy']
        
        # Calculate differences for each attribute value
        for attr_value in attribute_values:
            if (attr_value in model_results and 
                'accuracy' in model_results[attr_value] and 
                total_accuracy is not None):
                
                attr_accuracy = model_results[attr_value]['accuracy']
                difference = attr_accuracy - total_accuracy
                row.append(f"{difference * 100:+.1f}%")
            else:
                row.append("N/A")
        
        table_data.append(row)
    
    # Create DataFrame
    columns = ['Model'] + [f'"{value_name}"' for value_name in attribute_values]
    df = pd.DataFrame(table_data, columns=columns)
    df.set_index('Model', inplace=True)
    
    print(f"Differences to Total for attribute '{attribute}' (positive = better than total):")
    # Display based on preference
    if use_pandas:
        display(df)
    else:
        print(tabulate(df, headers=df.columns, tablefmt='grid', stralign='center'))