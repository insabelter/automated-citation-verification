import pandas as pd
from IPython.display import display
from scipy.stats import fisher_exact, chi2_contingency, kruskal, mannwhitneyu
import numpy as np
import sys
import os

# Add the parent directory to the path to import from scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from df_calculations import get_attribute_value_groups, eval_per_attribute_value

# ------------ Fisher's Exact Test ------------
def _perform_fisher_exact_test(subset_correct, subset_false, rest_correct, rest_false):
    """
    Helper function to perform Fisher's exact test.
    
    Args:
        subset_correct: Number of correct predictions in subset
        subset_false: Number of false predictions in subset
        rest_correct: Number of correct predictions in rest
        rest_false: Number of false predictions in rest
    
    Returns:
        Dictionary with Fisher's exact test results or None if test cannot be performed
    """
    contingency_table = [[subset_correct, subset_false],
                        [rest_correct, rest_false]]
    odds_ratio, p_value = fisher_exact(contingency_table)
    return {
        'odds_ratio': float(round(odds_ratio, 4)),
        'p_value': float(round(p_value, 4))
    }

def calc_fisher_exact_total_sub_unsub(attribute_subset_rest_results):
    results = {}

    subset_results = attribute_subset_rest_results['Subset']
    rest_results = attribute_subset_rest_results['Rest']
    
    # Calculate significance for Total dataset (combining both labels)
    total_subset_correct = subset_results['Substantiated']['True Classifications'] + subset_results['Unsubstantiated']['True Classifications']
    total_subset_false = subset_results['Substantiated']['False Classifications'] + subset_results['Unsubstantiated']['False Classifications']
    total_rest_correct = rest_results['Substantiated']['True Classifications'] + rest_results['Unsubstantiated']['True Classifications']
    total_rest_false = rest_results['Substantiated']['False Classifications'] + rest_results['Unsubstantiated']['False Classifications']

    total_result = _perform_fisher_exact_test(total_subset_correct, total_subset_false, total_rest_correct, total_rest_false)
    if total_result:
        results['Total'] = total_result
    
    # Calculate significance for Substantiated label
    sub_subset_correct = subset_results['Substantiated']['True Classifications']
    sub_subset_false = subset_results['Substantiated']['False Classifications']
    sub_rest_correct = rest_results['Substantiated']['True Classifications']
    sub_rest_false = rest_results['Substantiated']['False Classifications']

    sub_result = _perform_fisher_exact_test(sub_subset_correct, sub_subset_false, sub_rest_correct, sub_rest_false)
    if sub_result:
        results['Substantiated'] = sub_result
    
    # Calculate significance for Unsubstantiated label
    unsub_subset_correct = subset_results['Unsubstantiated']['True Classifications']
    unsub_subset_false = subset_results['Unsubstantiated']['False Classifications']
    unsub_rest_correct = rest_results['Unsubstantiated']['True Classifications']
    unsub_rest_false = rest_results['Unsubstantiated']['False Classifications']

    unsub_result = _perform_fisher_exact_test(unsub_subset_correct, unsub_subset_false, unsub_rest_correct, unsub_rest_false)
    if unsub_result:
        results['Unsubstantiated'] = unsub_result

    return results

def display_fishers_exact_test_results(significance_results):
    # Create table data
    table_data = []
    
    for attribute_value, test_results in significance_results.items():
        # Check if this has the new structure with label-specific results
        if 'Substantiated' in test_results or 'Unsubstantiated' in test_results or 'Total' in test_results:
            # Get Total results
            total_odds_ratio = "N/A"
            total_p_value = "N/A"
            if 'Total' in test_results:
                total_odds_ratio = f"{test_results['Total']['odds_ratio']:.4f}"
                total_p_value = f"{test_results['Total']['p_value']:.4f}"
            
            # Get Unsubstantiated results
            unsub_odds_ratio = "N/A"
            unsub_p_value = "N/A"
            if 'Unsubstantiated' in test_results:
                unsub_odds_ratio = f"{test_results['Unsubstantiated']['odds_ratio']:.4f}"
                unsub_p_value = f"{test_results['Unsubstantiated']['p_value']:.4f}"
            
            # Get Substantiated results
            sub_odds_ratio = "N/A"
            sub_p_value = "N/A"
            if 'Substantiated' in test_results:
                sub_odds_ratio = f"{test_results['Substantiated']['odds_ratio']:.4f}"
                sub_p_value = f"{test_results['Substantiated']['p_value']:.4f}"
            
            row = [
                attribute_value,
                total_odds_ratio,
                total_p_value,
                unsub_odds_ratio,
                unsub_p_value,
                sub_odds_ratio,
                sub_p_value
            ]
            table_data.append(row)

    # Create DataFrame
    columns = ['Attribute Value', 'Total Odds Ratio', 'Total P-value',
               'Unsubstantiated Odds Ratio', 'Unsubstantiated P-value', 
               'Substantiated Odds Ratio', 'Substantiated P-value']
    df = pd.DataFrame(table_data, columns=columns)
    df.set_index('Attribute Value', inplace=True)
    
    # Apply color styling to the DataFrame
    def color_p_values(val):
        """Color p-values based on significance level"""
        try:
            if val == "N/A":
                return ''
            p_val = float(val)
            if p_val <= 0.05:
                return 'background-color: darkgreen; color: white'
            elif p_val <= 0.2:
                return 'background-color: darkorange; color: white'
            else:
                return 'background-color: darkred; color: white'
        except (ValueError, TypeError):
            return ''
    
    # Apply styling only to p-value columns
    styled_df = df.style.map(color_p_values, subset=['Total P-value', 'Unsubstantiated P-value', 'Substantiated P-value'])
    display(styled_df)
    
    return df

# ------------ Chi-Squared Test ------------
def _perform_chi_squared_test(correct_counts, false_counts, attribute_values):
    """
    Helper function to perform chi-squared test and format results.
    
    Args:
        correct_counts: List of correct prediction counts for each attribute value
        false_counts: List of false prediction counts for each attribute value
        attribute_values: List of attribute values corresponding to the counts
    
    Returns:
        Dictionary with chi-squared test results or error information
    """
    if len(correct_counts) >= 2 and any(false_counts) and any(correct_counts):
        contingency_table = [correct_counts, false_counts]
        try:
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Map expected frequencies to attribute values for better understanding
            expected_dict = {}
            for i, attr_value in enumerate(attribute_values):
                expected_dict[attr_value] = {
                    'expected_correct': expected[0][i],
                    'expected_false': expected[1][i]
                }
            
            return {
                'chi2_statistic': float(round(chi2_stat, 4)),
                'p_value': float(round(p_value, 4)),
                'degrees_of_freedom': dof,
                'expected_frequencies': expected_dict
            }
        except ValueError as e:
            return {'error': str(e)}
    
    return None

def calc_chi_squared_total_sub_unsub(attribute_results, attribute_values):
    """
    Performs chi-squared tests on contingency tables for Total, Substantiated, and Unsubstantiated predictions
    across different attribute values.
    
    Args:
        attribute_results: Dictionary with structure {attribute_value: {eval_predictions results...}}
        attribute_values: List of attribute values to include in the test
    
    Returns:
        Dictionary with chi-squared test results for Total, Substantiated, and Unsubstantiated
    """
    results = {}
    
    if len(attribute_values) < 2:
        return results  # Need at least 2 groups for chi-squared test
    
    # Prepare data for Total dataset (combining both labels)
    total_correct = []
    total_false = []
    
    # Prepare data for Substantiated label
    sub_correct = []
    sub_false = []
    
    # Prepare data for Unsubstantiated label  
    unsub_correct = []
    unsub_false = []
    
    # Extract data for each attribute value
    for attr_value in attribute_values:
        if attr_value in attribute_results:
            results_for_value = attribute_results[attr_value]
            
            # Total correct and false predictions (combining both labels)
            total_correct_count = (results_for_value['Substantiated']['True Classifications'] + 
                                 results_for_value['Unsubstantiated']['True Classifications'])
            total_false_count = (results_for_value['Substantiated']['False Classifications'] + 
                               results_for_value['Unsubstantiated']['False Classifications'])
            
            total_correct.append(total_correct_count)
            total_false.append(total_false_count)
            
            # Substantiated predictions
            sub_correct.append(results_for_value['Substantiated']['True Classifications'])
            sub_false.append(results_for_value['Substantiated']['False Classifications'])
            
            # Unsubstantiated predictions
            unsub_correct.append(results_for_value['Unsubstantiated']['True Classifications'])
            unsub_false.append(results_for_value['Unsubstantiated']['False Classifications'])
    
    # Perform chi-squared tests using helper function
    total_result = _perform_chi_squared_test(total_correct, total_false, attribute_values)
    if total_result:
        results['Total'] = total_result
    
    sub_result = _perform_chi_squared_test(sub_correct, sub_false, attribute_values)
    if sub_result:
        results['Substantiated'] = sub_result
    
    unsub_result = _perform_chi_squared_test(unsub_correct, unsub_false, attribute_values)
    if unsub_result:
        results['Unsubstantiated'] = unsub_result
    
    return results

def display_chi_squared_test_results(significance_results):
    """
    Display chi-squared test results in a formatted table with color-coded p-values.
    
    Args:
        significance_results: Dictionary with structure {category: {test_results...}} 
                             where category is 'Total', 'Substantiated', or 'Unsubstantiated'
    
    Returns:
        pandas.DataFrame: The formatted results table
    """
    # Get Total results
    total_chi2 = "N/A"
    total_p_value = "N/A"
    if 'Total' in significance_results and 'error' not in significance_results['Total']:
        total_chi2 = f"{significance_results['Total']['chi2_statistic']:.4f}"
        total_p_value = f"{significance_results['Total']['p_value']:.4f}"
    
    # Get Unsubstantiated results
    unsub_chi2 = "N/A"
    unsub_p_value = "N/A"
    if 'Unsubstantiated' in significance_results and 'error' not in significance_results['Unsubstantiated']:
        unsub_chi2 = f"{significance_results['Unsubstantiated']['chi2_statistic']:.4f}"
        unsub_p_value = f"{significance_results['Unsubstantiated']['p_value']:.4f}"
    
    # Get Substantiated results
    sub_chi2 = "N/A"
    sub_p_value = "N/A"
    if 'Substantiated' in significance_results and 'error' not in significance_results['Substantiated']:
        sub_chi2 = f"{significance_results['Substantiated']['chi2_statistic']:.4f}"
        sub_p_value = f"{significance_results['Substantiated']['p_value']:.4f}"
    
    # Create single row of data
    table_data = [[
        total_chi2,
        total_p_value,
        unsub_chi2,
        unsub_p_value,
        sub_chi2,
        sub_p_value
    ]]

    # Create DataFrame
    columns = ['Total Chi2', 'Total P-value',
               'Unsubstantiated Chi2', 'Unsubstantiated P-value', 
               'Substantiated Chi2', 'Substantiated P-value']
    df = pd.DataFrame(table_data, columns=columns)
    
    # Apply color styling to the DataFrame
    def color_p_values(val):
        """Color p-values based on significance level"""
        try:
            if val == "N/A":
                return ''
            p_val = float(val)
            if p_val <= 0.05:
                return 'background-color: darkgreen; color: white'
            elif p_val <= 0.2:
                return 'background-color: darkorange; color: white'
            else:
                return 'background-color: darkred; color: white'
        except (ValueError, TypeError):
            return ''
    
    # Apply styling only to p-value columns
    styled_df = df.style.map(color_p_values, subset=['Total P-value', 'Unsubstantiated P-value', 'Substantiated P-value'])
    display(styled_df)
    
    return df

# ------------ Permutation Test -------------
def _extract_metric_from_results(results, metric_path):
    """
    Helper function to extract a specific metric from eval_predictions results.
    
    Args:
        results: Results dictionary from eval_predictions_per_attribute_value
        metric_path: List describing path to metric (e.g., ['Accuracy'] or ['Substantiated', 'Recall'])
    
    Returns:
        Metric value or None if not found
    """
    metric_value = results
    for key in metric_path:
        if key in metric_value:
            metric_value = metric_value[key]
        else:
            return None
    return metric_value

def calc_permutation_test_total_sub_unsub(df, attribute, group_numbers_from=False, n_permutations=1000, seed=42):
    """
    Performs two-sided permutation tests for the difference of classification metric values between groups.
    Efficiently calculates all metrics (Total, Substantiated, Unsubstantiated) in a single permutation loop.
    
    Args:
        df: DataFrame with prediction results
        attribute: Column name of the attribute to shuffle
        group_numbers_from: Whether to group attribute values by their numeric part
        n_permutations: Number of permutations to perform
        seed: Random seed for reproducible results
    
    Returns:
        Dictionary with all permutation test results for Total, Substantiated, and Unsubstantiated metrics
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate observed results per attribute value
    attribute_groups = get_attribute_value_groups(df, attribute, group_numbers_from)
    observed_results = eval_per_attribute_value(df, attribute, attribute_groups)
    attribute_values = [val for val in observed_results.keys() if val != 'Total']
    
    if len(attribute_values) < 2:
        return {}
    
    # Define all metrics to test
    metrics_to_test = {
        'Total': {
            'Accuracy': ['Accuracy'],
            'Balanced Accuracy': ['Balanced Accuracy']
        },
        'Substantiated': {
            'Precision': ['Substantiated', 'Precision'],
            'Recall': ['Substantiated', 'Recall'],
            'F1 Score': ['Substantiated', 'F1 Score']
        },
        'Unsubstantiated': {
            'Precision': ['Unsubstantiated', 'Precision'],
            'Recall': ['Unsubstantiated', 'Recall'],
            'F1 Score': ['Unsubstantiated', 'F1 Score']
        }
    }
    
    # Extract observed values for all metrics
    observed_metrics = {}
    for category, metric_dict in metrics_to_test.items():
        observed_metrics[category] = {}
        for metric_name, metric_path in metric_dict.items():
            observed_values = []
            for attr_value in attribute_values:
                if attr_value in observed_results:
                    metric_value = _extract_metric_from_results(observed_results[attr_value], metric_path)
                    if metric_value is not None:
                        observed_values.append(metric_value)
                    else:
                        observed_values = None
                        break
            
            if observed_values is not None:
                observed_diff = np.var(observed_values)
                observed_metrics[category][metric_name] = {
                    'observed_values': observed_values,
                    'observed_difference': observed_diff,
                    'permuted_differences': [],
                    'extreme_count': 0
                }
    
    # Perform permutations once and calculate all metrics
    df_copy = df.copy()
    
    for _ in range(n_permutations):
        # Shuffle the attribute values once
        shuffled_attributes = np.random.permutation(df_copy[attribute].values)
        df_copy[attribute] = shuffled_attributes
        
        # Recalculate metrics with shuffled attributes once
        permuted_attribute_groups = get_attribute_value_groups(df_copy, attribute, group_numbers_from)
        permuted_results = eval_per_attribute_value(df_copy, attribute, permuted_attribute_groups)
        
        # Extract all metrics from this single permutation
        for category, metric_dict in observed_metrics.items():
            for metric_name, metric_data in metric_dict.items():
                metric_path = metrics_to_test[category][metric_name]
                
                # Extract permuted metric values
                permuted_values = []
                for attr_value in attribute_values:
                    if attr_value in permuted_results:
                        metric_value = _extract_metric_from_results(permuted_results[attr_value], metric_path)
                        if metric_value is not None:
                            permuted_values.append(metric_value)
                        else:
                            permuted_values.append(0)  # Default if metric not found
                
                # Calculate permuted difference
                permuted_diff = np.var(permuted_values)
                
                # Store permuted difference
                metric_data['permuted_differences'].append(permuted_diff)
                
                # Count extreme values (two-sided test)
                if permuted_diff >= metric_data['observed_difference']:
                    metric_data['extreme_count'] += 1
    
    # Calculate final results for all metrics
    results = {}
    for category, metric_dict in observed_metrics.items():
        if metric_dict:  # Only add category if it has valid metrics
            results[category] = {}
            for metric_name, metric_data in metric_dict.items():
                p_value = metric_data['extreme_count'] / n_permutations
                average_difference = np.mean(metric_data['permuted_differences'])
                
                results[category][metric_name] = {
                    'p_value': p_value,
                    'observed_variance': float(round(metric_data['observed_difference'], 4)),
                    'average_variance': float(round(average_difference, 4)),
                    'difference of variances': float(round(metric_data['observed_difference'], 4) - float(round(average_difference, 4))),
                    'n_permutations': n_permutations,
                    'n_groups': len(attribute_values),
                    'observed_values': metric_data['observed_values']
                }
    
    return results

def display_permutation_test_results(significance_results):
    """
    Display permutation test results in a formatted table with color-coded p-values.
    
    Args:
        significance_results: Dictionary with structure {category: {metric_name: {test_results...}}}
                             where category is 'Total', 'Substantiated', or 'Unsubstantiated'
    
    Returns:
        pandas.DataFrame: The formatted results table
    """
    # Define the desired row order
    metric_order = ['Balanced Accuracy', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Collect all unique metric names across all categories
    all_metrics = set()
    for category_results in significance_results.values():
        for metric_name in category_results.keys():
            all_metrics.add(metric_name)
    
    # Use only metrics that exist in our data, in the specified order
    sorted_metrics = [metric for metric in metric_order if metric in all_metrics]
    
    # Create table data
    table_data = []
    
    for metric_name in sorted_metrics:
        row = [metric_name]
        
        # Add Total columns (Var Diff and P-value)
        if 'Total' in significance_results and metric_name in significance_results['Total']:
            total_results = significance_results['Total'][metric_name]
            row.append(f"{total_results['difference of variances']:.6f}")
            row.append(f"{total_results['p_value']:.4f}")
        else:
            row.extend(["N/A", "N/A"])
        
        # Add Unsubstantiated columns (Var Diff and P-value)
        if 'Unsubstantiated' in significance_results and metric_name in significance_results['Unsubstantiated']:
            unsub_results = significance_results['Unsubstantiated'][metric_name]
            row.append(f"{unsub_results['difference of variances']:.6f}")
            row.append(f"{unsub_results['p_value']:.4f}")
        else:
            row.extend(["N/A", "N/A"])
        
        # Add Substantiated columns (Var Diff and P-value)
        if 'Substantiated' in significance_results and metric_name in significance_results['Substantiated']:
            sub_results = significance_results['Substantiated'][metric_name]
            row.append(f"{sub_results['difference of variances']:.6f}")
            row.append(f"{sub_results['p_value']:.4f}")
        else:
            row.extend(["N/A", "N/A"])
        
        table_data.append(row)
    
    # Create DataFrame
    columns = [
        'Metric',
        'Total Var Diff', 'Total P-value',
        'Unsubstantiated Var Diff', 'Unsubstantiated P-value',
        'Substantiated Var Diff', 'Substantiated P-value'
    ]
    df = pd.DataFrame(table_data, columns=columns)
    df.set_index('Metric', inplace=True)
    
    # Apply color styling to the DataFrame
    def color_p_values(val):
        """Color p-values based on significance level"""
        try:
            if val == "N/A":
                return ''
            p_val = float(val)
            if p_val <= 0.05:
                return 'background-color: darkgreen; color: white'
            elif p_val <= 0.2:
                return 'background-color: darkorange; color: white'
            else:
                return 'background-color: darkred; color: white'
        except (ValueError, TypeError):
            return ''
    
    # Apply styling only to p-value columns
    p_value_columns = ['Total P-value', 'Substantiated P-value', 'Unsubstantiated P-value']
    styled_df = df.style.map(color_p_values, subset=p_value_columns)
    display(styled_df)
    
    return df

# ------------ Extract only p-values from all results ------------
def extract_p_values(significance_tests_results):
    """
    Extract only p-values from the significance tests results dictionary.
    
    Parameters:
    significance_tests_results: Dictionary containing significance test results with structure:
                               {attribute: {test_type: {group/label: {metric: {results...}}}}}
    
    Returns:
    Dictionary with the same structure but containing only p-values
    """
    p_values_only = {}
    
    for attribute, tests in significance_tests_results.items():
        p_values_only[attribute] = {}
        
        # Extract Fisher Exact p-values
        if 'Fisher Exact' in tests:
            p_values_only[attribute]['Fisher Exact'] = {}
            for group, labels in tests['Fisher Exact'].items():
                p_values_only[attribute]['Fisher Exact'][group] = {}
                for label, results in labels.items():
                    p_values_only[attribute]['Fisher Exact'][group][label] = float(results['p_value'])
        
        # Extract Chi-Squared p-values
        if 'Chi-Squared' in tests:
            p_values_only[attribute]['Chi-Squared'] = {}
            for label, results in tests['Chi-Squared'].items():
                p_values_only[attribute]['Chi-Squared'][label] = float(results['p_value'])
        
        # Extract Permutation Test p-values
        if 'Permutation Test' in tests:
            p_values_only[attribute]['Permutation Test'] = {}
            for label, metrics in tests['Permutation Test'].items():
                p_values_only[attribute]['Permutation Test'][label] = {}
                for metric, results in metrics.items():
                    p_values_only[attribute]['Permutation Test'][label][metric] = float(results['p_value'])
    
    return p_values_only