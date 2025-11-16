from tabulate import tabulate
import pandas as pd
from IPython.display import display

def eval_predictions(df, include_relabelled_partially=True, include_not_originally_downloaded=True):
    G_Total = 0
    Sub_Correct_Total = 0
    Unsub_Correct_Total = 0
    Sub_True = 0
    Sub_False = 0
    Unsub_True = 0
    Unsub_False = 0

    invalid_labels = {
        'Unsubstantiated': [],
        'Substantiated': []
    }

    for index, row in df.iterrows():
        if row['Reference Article Downloaded'] == 'Yes':
            if include_not_originally_downloaded or row['Reference Article PDF Available'] == 'Yes':
                if include_relabelled_partially or row['Previously Partially Substantiated'] != 'x':
                    G_Total += 1
                    target_label = row['Label']
                    model_label = row['Model Classification Label']

                    assert target_label in ['Unsubstantiated', 'Substantiated'], f"Row {index} Label is not a valid label: {target_label}"

                    invalid_label = False
                    if model_label not in ['Unsubstantiated', 'Substantiated']:
                        invalid_labels[target_label].append(model_label)
                        print(f"Row {index} Model Classification Label is not a valid label: {model_label}")
                        invalid_label = True
                
                    if target_label == 'Substantiated':
                        Sub_Correct_Total += 1
                        if model_label == 'Substantiated':
                            Sub_True += 1
                        elif model_label == 'Unsubstantiated':
                            Sub_False += 1
                        elif invalid_label:
                            Sub_False += 1
                    elif target_label == 'Unsubstantiated':
                        Unsub_Correct_Total += 1
                        if model_label == 'Substantiated':
                            Unsub_False += 1
                        elif model_label == 'Unsubstantiated':
                            Unsub_True += 1
                        elif invalid_label:
                            Unsub_False += 1

    assert G_Total == Sub_Correct_Total + Unsub_Correct_Total, f"Total G ({G_Total}) does not equal Sub_Correct_Total ({Sub_Correct_Total}) + Unsub_Correct_Total ({Unsub_Correct_Total})"
    assert Sub_True + Sub_False == Sub_Correct_Total, f"Sub_True ({Sub_True}) + Unsub_False ({Unsub_False}) does not equal Sub_Correct_Total ({Sub_Correct_Total})"
    assert Unsub_True + Unsub_False == Unsub_Correct_Total, f"Unsub_True ({Unsub_True}) + Sub_False ({Sub_False}) does not equal Unsub_Correct_Total ({Unsub_Correct_Total})"

    results = {
        'Total': G_Total,
        'Substantiated': {
            'Label Total': Sub_Correct_Total,
            'True Classifications': Sub_True,
            'False Classifications': Sub_False,
            'Precision': round(Sub_True / (Sub_True + Unsub_False) if (Sub_True + Unsub_False) > 0 else 0.0, 3),
            'Recall': round(Sub_True / Sub_Correct_Total if Sub_Correct_Total > 0 else 0.0, 3),
            'F1 Score': 0,
            'Invalid_Labels': invalid_labels['Substantiated']
        },
        'Unsubstantiated': {
            'Label Total': Unsub_Correct_Total,
            'True Classifications': Unsub_True,
            'False Classifications': Unsub_False,
            'Precision': round(Unsub_True / (Unsub_True + Sub_False) if (Unsub_True + Sub_False) > 0 else 0.0, 3),
            'Recall': round(Unsub_True / Unsub_Correct_Total if Unsub_Correct_Total > 0 else 0.0, 3),
            'F1 Score': 0,
            'Invalid_Labels': invalid_labels['Unsubstantiated']
        },
        'Accuracy': round((Sub_True + Unsub_True) / G_Total if G_Total > 0 else 0.0, 3),
        'Error Rate': round((Sub_False + Unsub_False) / G_Total if G_Total > 0 else 0.0, 3),
        'Balanced Accuracy': 0
    }
    results['Substantiated']['F1 Score'] = round(2 * ((results['Substantiated']['Precision'] * results['Substantiated']['Recall']) / (results['Substantiated']['Precision'] + results['Substantiated']['Recall'])) if (results['Substantiated']['Precision'] + results['Substantiated']['Recall']) > 0 else 0.0, 3)
    results['Unsubstantiated']['F1 Score'] = round(2 * ((results['Unsubstantiated']['Precision'] * results['Unsubstantiated']['Recall']) / (results['Unsubstantiated']['Precision'] + results['Unsubstantiated']['Recall'])) if (results['Unsubstantiated']['Precision'] + results['Unsubstantiated']['Recall']) > 0 else 0.0, 3)

    results['Balanced Accuracy'] = round((results['Substantiated']['Recall'] + results['Unsubstantiated']['Recall']) / 2, 3)

    return results
    
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

def display_model_results_table(model_results_dict):
    # Create a list to store the DataFrame data
    df_data = []
    
    for model_name, results in model_results_dict.items():
        row = {'Model': model_name}
        
        # Extract metrics from the new schema
        row['Accuracy'] = results.get('Accuracy', None)
        row['Balanced Accuracy'] = results.get('Balanced Accuracy', None)
        row['Precision (Unsubstantiated)'] = results.get('Unsubstantiated', {}).get('Precision', None)
        row['Recall (Unsubstantiated)'] = results.get('Unsubstantiated', {}).get('Recall', None)
        row['F1 Score (Unsubstantiated)'] = results.get('Unsubstantiated', {}).get('F1 Score', None)
        row['Precision (Substantiated)'] = results.get('Substantiated', {}).get('Precision', None)
        row['Recall (Substantiated)'] = results.get('Substantiated', {}).get('Recall', None)
        row['F1 Score (Substantiated)'] = results.get('Substantiated', {}).get('F1 Score', None)
        
        df_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    df.set_index('Model', inplace=True)
    
    # Display the DataFrame
    display(df)

# def get_preds_results(results):
#     return {
#         "Unsubstantiated": { # negative class
#             "preds": results['TN'] + results['FN'],
#             "correct_preds": results['TN'],
#             "correct_total": results['N (Unsubstantiated)'],
#         },
#         "Substantiated": { # positive class
#             "preds": results['TP'] + results['FP'],
#             "correct_preds": results['TP'],
#             "correct_total": results['P (Substantiated)'],
#         },
#     }

def eval_predictions_per_attribute_value(df, attribute, include_relabelled_partially, group_numbers_from=False):
    results = {}
    results['Total'] = eval_predictions(df, include_relabelled_partially=include_relabelled_partially)
    
    # Special case for "Claim Contains Number or Formula" attribute
    if attribute == "Claim Contains Number or Formula" and group_numbers_from == "Number/Formula":
        # Group 'No' separately
        df_no = df[df[attribute] == 'No']
        results['No'] = eval_predictions(df_no, include_relabelled_partially=include_relabelled_partially)
        
        # Group 'Number' and 'Formula' together
        df_number_formula = df[df[attribute].isin(['Number', 'Formula'])]
        results['Number/\nFormula'] = eval_predictions(df_number_formula, include_relabelled_partially=include_relabelled_partially)
        
        return results
    
    # Get unique attribute values and sort them
    attribute_values = df[attribute].unique()
    
    # Try to sort numerically first, fall back to alphabetical sorting
    try:
        # Attempt to convert to numbers and sort numerically
        sorted_values = sorted(attribute_values, key=lambda x: float(x))
        values_are_numeric = True
    except (ValueError, TypeError):
        # If conversion fails, sort alphabetically
        sorted_values = sorted(attribute_values, key=lambda x: str(x))
        values_are_numeric = False
    
    # If group_numbers_from is specified and values are numeric, group values
    if group_numbers_from is not False and values_are_numeric:
        # Process values below the threshold individually
        for value in sorted_values:
            numeric_value = float(value)
            if numeric_value < group_numbers_from:
                df_value = df[df[attribute] == value]
                results_value = eval_predictions(df_value, include_relabelled_partially=include_relabelled_partially)
                results[str(value)] = results_value
        
        # Group values >= threshold together
        values_to_group = [v for v in sorted_values if float(v) >= group_numbers_from]
        if values_to_group:
            # Create a combined dataframe for all values >= threshold using isin
            df_grouped = df[df[attribute].isin(values_to_group)]
            results_grouped = eval_predictions(df_grouped, include_relabelled_partially=include_relabelled_partially)
            results[f"≥ {group_numbers_from}"] = results_grouped
    else:
        # Process all values individually (original behavior)
        for value in sorted_values:
            df_value = df[df[attribute] == value]
            results_value = eval_predictions(df_value, include_relabelled_partially=include_relabelled_partially)
            results[str(value)] = results_value
    
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

# Evaluate per attribute value 
def eval_per_attribute_value(df, attribute, attribute_values_per_group):
    # attribute_groups: [('1', [1]), ('2', [2]), ('>= 3', [3, 4, 5, 6, 7, 8])]
    results = {}
    for group_name, attribute_values in attribute_values_per_group:
        group_df = df[df[attribute].isin(attribute_values)]
        results[group_name] = eval_predictions(group_df)
    return results

def eval_attribute_subset_vs_rest(df, attribute, attribute_values):
    # Create subset and rest DataFrames
    subset_df = df[df[attribute].isin(attribute_values)]
    rest_df = df[~df[attribute].isin(attribute_values)]
    
    subset_results = eval_predictions(subset_df, include_relabelled_partially=True)
    rest_results = eval_predictions(rest_df, include_relabelled_partially=True)
    
    return {
        'Subset': subset_results,
        'Rest': rest_results,
    }

def get_attribute_value_groups(df, attribute, group_numbers_from=False):
    # Special case for "Claim Contains Number or Formula" attribute
    if attribute == "Claim Contains Number or Formula" and group_numbers_from == "Number/Formula":
        return [('No', ['No']), ('Number/\nFormula', ['Number', 'Formula'])]
    
    attribute_values = df[attribute].unique()
    
    # Try to sort numerically first, fall back to alphabetical sorting
    try:
        # Attempt to convert to numbers and sort numerically
        sorted_values = sorted(attribute_values, key=lambda x: float(x))
        values_are_numeric = True
    except (ValueError, TypeError):
        # If conversion fails, sort alphabetically
        sorted_values = sorted(attribute_values, key=lambda x: str(x))
        values_are_numeric = False
    
    groups = []
    
    # If group_numbers_from is specified and values are numeric, group values
    if group_numbers_from is not False and values_are_numeric:
        # Process values below threshold individually
        for value in sorted_values:
            if float(value) < group_numbers_from:
                groups.append((str(value), [value]))
        
        # Group values >= threshold together
        values_to_group = [value for value in sorted_values if float(value) >= group_numbers_from]
        if values_to_group:
            group_name = f"≥ {group_numbers_from}"
            groups.append((group_name, values_to_group))
    else:
        # Process all values individually (original behavior)
        for attribute_value in attribute_values:
            groups.append((str(attribute_value), [attribute_value]))
    
    return groups
