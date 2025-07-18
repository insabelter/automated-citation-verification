import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np

citation_functions = {
    "Use": ["Apply", "Extend"],
    "Compare": ["Criticize", "Contrast", "Confirm"],
    "Related": ["Definition/Proof", "Fundamentals", "Acknowledge"],
    "Background": ["Introduction/Bigger picture", "Unrelated/Unclear"]
}

def show_distribution(df, column_name, include_nan=True, sorting=None):
    # Count the occurrences of each source, including NaN values if specified
    source_counts = df[column_name].value_counts(dropna=(not include_nan))

    # Replace NaN with a string label for visualization
    if include_nan:
        source_counts.index = source_counts.index.fillna('NaN')

    if sorting:
        source_counts = source_counts.reindex(sorting, fill_value=0)

    # Plot the bar diagram
    plt.figure(figsize=(10, 6))
    ax = source_counts.plot(kind='bar')
    plt.title('Distribution of column: ' + column_name)
    plt.xlabel(column_name)
    plt.ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Add the total number above each bar, aligned vertically with the middle of the bar
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height() + (p.get_width()*0.5)), 
                    ha='center', va='center')

    plt.show()

def show_distribution_pie(df, column_name, include_nan=True, sorting=None):
    # Count the occurrences of each source, including NaN values if specified
    source_counts = df[column_name].value_counts(dropna=(not include_nan))

    # Replace NaN with a string label for visualization
    if include_nan:
        source_counts.index = source_counts.index.fillna('NaN')

    if sorting:
        source_counts = source_counts.reindex(sorting, fill_value=0)

    # Format labels to include the count after the label name
    labels = [f"{label} ({count})" for label, count in zip(source_counts.index, source_counts)]

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(source_counts, labels=labels, autopct='%1.1f%%', startangle=90, counterclock=False)
    plt.title('Distribution of column: ' + column_name)
    plt.show()

def show_distribution_comparison(df1, df2, column_name, df1_name="Full Data", df2_name="False Predicted Data", include_nan=True):
    # Count the occurrences of each value in the specified column for both dataframes, including NaN values
    source_counts_df1 = df1[column_name].value_counts(dropna=(not include_nan))
    source_counts_df2 = df2[column_name].value_counts(dropna=(not include_nan))

    # Replace NaN with a string label for visualization
    if include_nan:
        source_counts_df1.index = source_counts_df1.index.fillna('NaN')
        source_counts_df2.index = source_counts_df2.index.fillna('NaN')

    # Create subplots for side-by-side bar charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Plot the first dataframe
    ax1 = source_counts_df1.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title(f'Distribution ({column_name}): {df1_name}')
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel('Count')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Add the total number above each bar for the first dataframe
    for p in ax1.patches:
        ax1.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height() + 1), 
                     ha='center', va='center')

    # Plot the second dataframe
    ax2 = source_counts_df2.plot(kind='bar', ax=axes[1], color='salmon')
    axes[1].set_title(f'Distribution ({column_name}): {df2_name}')
    axes[1].set_xlabel(column_name)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    # Add the total number above each bar for the second dataframe
    for p in ax2.patches:
        ax2.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height() + 1), 
                     ha='center', va='center')

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

def show_distribution_comparison_pie(df1, df2, column_name, df1_name="Full Data", df2_name="False Predicted Data", include_nan=True, label_threshold_percent=5):
    # Count the occurrences of each value in the specified column for both dataframes, including NaN values
    source_counts_df1 = df1[column_name].value_counts(dropna=(not include_nan))
    source_counts_df2 = df2[column_name].value_counts(dropna=(not include_nan))

    # Replace NaN with a string label for visualization
    if include_nan:
        source_counts_df1.index = source_counts_df1.index.fillna('NaN')
        source_counts_df2.index = source_counts_df2.index.fillna('NaN')

    # Ensure consistent colors for the same categories
    all_categories = set(source_counts_df1.index).union(set(source_counts_df2.index))
    colors = plt.cm.tab20.colors[:len(all_categories)]
    color_map = {category: colors[i] for i, category in enumerate(all_categories)}

    # Helper function to filter labels based on percentage threshold and truncate long labels
    def filter_labels_and_percentages(counts, total, label_threshold_percent):
        labels = []
        percentages = []
        for label, count in counts.items():
            percentage = (count / total) * 100
            if round(percentage, 1) >= label_threshold_percent:
                truncated_label = (label[:47] + '...') if len(label) > 50 else label
                labels.append(truncated_label)
            else:
                labels.append('')  # Empty label for entries below threshold
            percentages.append(percentage)
        return labels, percentages

    # Create subplots for side-by-side pie charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot the first dataframe
    total_df1 = source_counts_df1.sum()
    labels_df1, percentages_df1 = filter_labels_and_percentages(source_counts_df1, total_df1, label_threshold_percent)
    wedges, texts, autotexts = axes[0].pie(
        source_counts_df1, labels=labels_df1, autopct='%1.1f%%', startangle=90, counterclock=False,
        colors=[color_map[cat] for cat in source_counts_df1.index], pctdistance=0.85
    )
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
    axes[0].set_title(f'Distribution ({column_name}): {df1_name}')

    # Plot the second dataframe
    total_df2 = source_counts_df2.sum()
    labels_df2, percentages_df2 = filter_labels_and_percentages(source_counts_df2, total_df2, label_threshold_percent)
    wedges, texts, autotexts = axes[1].pie(
        source_counts_df2, labels=labels_df2, autopct='%1.1f%%', startangle=90, counterclock=False,
        colors=[color_map[cat] for cat in source_counts_df2.index], pctdistance=0.85
    )
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
    axes[1].set_title(f'Distribution ({column_name}): {df2_name}')

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

def show_retracted_distribution_pie(df):
    # Create a new column to categorize the rows
    df['Retracted Status'] = df.apply(lambda row: 'Reference and Citing' if row['Citing Article Retracted'] == 'Yes' and row['Reference Article Retracted'] == 'Yes' else 
                                    ('Citing Only' if row['Citing Article Retracted'] == 'Yes' else 
                                    ('Reference Only' if row['Reference Article Retracted'] == 'Yes' else 'None Retracted')), axis=1)

    # Count the occurrences of each category
    retracted_counts = df['Retracted Status'].value_counts()

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(retracted_counts, labels=retracted_counts.index, autopct='%1.1f%%', startangle=90, counterclock=False)
    plt.title('Distribution of Retracted Articles')
    plt.show()

def show_source_distribution_citing_article_retracted(df):
    # Calculate the total count and the count of 'Yes' for 'Citing Article Retracted' for each source
    source_total_counts = df['Source'].value_counts()
    source_retracted_counts = df[df['Citing Article Retracted'] == 'Yes']['Source'].value_counts()

    # Calculate the percentage of 'Yes' for 'Citing Article Retracted'
    source_retracted_percentage = (source_retracted_counts / source_total_counts) * 100

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Total': source_total_counts,
        'Retracted': source_retracted_counts,
        'Retracted Percentage': source_retracted_percentage
    }).fillna(0)

    # Plot the bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(plot_data.index, plot_data['Total'], label='Total')
    plt.bar(plot_data.index, plot_data['Retracted'], label='Retracted')

    # Add the percentage labels
    for bar, percentage in zip(bars, plot_data['Retracted Percentage']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{percentage:.1f}%', ha='center', va='bottom')

    plt.title('Distribution of Sources with Citing Article Retracted Percentage')
    plt.xlabel('Source')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()

def show_citation_function_main_distribution_pie(df, show_totals=False):
    # Strip whitespaces from the "Citation Function: Main" column
    df = df[df["Citation Function: Main"].notna()].copy()
    df["Citation Function: Main"] = df["Citation Function: Main"].str.strip()

    # Count the occurrences of each source, including NaN values if specified
    source_counts = df["Citation Function: Main"].value_counts()

    # Reindex source_counts to match the custom order
    source_counts = source_counts.reindex(citation_functions.keys())

    # Helper function to format labels with totals if required
    def format_label(label, count):
        return f"{label} ({count})" if show_totals else label

    # Format labels with totals if the parameter is set to True
    labels = [format_label(label, count) for label, count in zip(source_counts.index, source_counts)]

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(source_counts, labels=labels, autopct='%1.1f%%', startangle=90, counterclock=False)
    plt.title('Distribution of column: ' + "Citation Function: Main")
    plt.show()

def get_color(sub_function):
    # Define colors for each main category
    main_category_colors = {
        "Use": "#1f77b4",  # Blue
        "Compare": "#ff7f0e",  # Orange
        "Related": "#2ca02c",  # Green
        "Background": "#d62728"  # Red
    }

    # Generate sub-category colors based on main category colors
    sub_function_colors = {}
    for main_category, sub_categories in citation_functions.items():
        base_color = main_category_colors[main_category]
        for i, sub_category in enumerate(sub_categories):
            # Adjust lightness for sub-categories
            lightness_factor = 1 - (i * 0.2)
            rgba = plt.cm.colors.to_rgba(base_color)
            adjusted_rgba = tuple(channel * lightness_factor if index < 3 else channel for index, channel in enumerate(rgba))
            sub_function_colors[sub_category] = plt.cm.colors.to_hex(adjusted_rgba)
            
    if sub_function in sub_function_colors:
        return sub_function_colors[sub_function]
    else:
        print(f"Warning: {sub_function} not found in sub_function_colors.")
        return "#000000"  # Default color (black) for unknown categories
    
def show_citation_function_sub_distribution_pie(df, show_totals=False):
    sub_functions = [sub for sub_list in citation_functions.values() for sub in sub_list]

    # Strip whitespaces from the "Citation Function: Sub" column
    df = df[df["Citation Function: Sub"].notna()].copy()
    df["Citation Function: Sub"] = df["Citation Function: Sub"].str.strip()

    # Count the occurrences of each source, including NaN values if specified
    source_counts = df["Citation Function: Sub"].value_counts()

    # Reindex source_counts to match the custom order
    source_counts = source_counts.reindex(sub_functions, fill_value=0)

    # Map the colors to the sub-functions
    colors = [get_color(sub_function) for sub_function in source_counts.index]

    # Helper function to format labels with totals if required
    def format_label(label, count):
        return f"{label} ({count})" if show_totals else label

    # Format labels with totals if the parameter is set to True
    labels = [format_label(label, count) for label, count in zip(source_counts.index, source_counts)]

    # Plot the pie chart with defined colors
    plt.figure(figsize=(8, 8))
    plt.pie(source_counts, labels=labels, autopct='%1.1f%%', startangle=90, counterclock=False, colors=colors)
    plt.title('Distribution of column: ' + "Citation Function: Sub")
    plt.show()

def show_preds_vs_correct_preds_vs_total(data_dicts, titles, title="Comparison of Accuracies, Predictions, Correct Predictions and Correct Totals per Label across Datasets", labels=['unsubstantiate', 'fully substantiate'], smaller_figures=False):
    """
    Plot the distribution of predictions, correct predictions, and total label counts for each dictionary.

    Each dictionary should have the following structure:
    {
        "unsubstantiate": {
            "preds": ...,
            "correct_preds": ...,
            "correct_total": ...,
        },
        "fully substantiate": {
            "preds": ...,
            "correct_preds": ...,
            "correct_total": ...,
        },
    }
    """
    # Colors for the bars
    colors = {labels[0]: 'red', labels[1]: 'green'}

    # Create subplots
    if smaller_figures:
        fig, axes = plt.subplots(1, len(data_dicts), figsize=(10, 6), sharey=True)
        fig.suptitle(title, fontsize=12)
    else:
        fig, axes = plt.subplots(1, len(data_dicts), figsize=(18, 6), sharey=True)
        fig.suptitle(title, fontsize=16)


    # Fixed bar width
    bar_width = 0.6

    # Plot each dictionary
    for ax, data, title in zip(axes, data_dicts, titles):
        labels = list(data.keys())
        total_preds = [data[label]['preds'] for label in labels]
        correct_preds = [data[label]['correct_preds'] for label in labels]
        correct_totals = [data[label]['correct_total'] for label in labels]

        # Calculate total accuracy for this dataset
        total_correct_preds = sum(correct_preds)
        total_correct_totals = sum(correct_totals)
        total_accuracy = total_correct_preds / total_correct_totals if total_correct_totals > 0 else 0

        # Plot total predictions
        bars_total = ax.bar(labels, total_preds, color=[colors[label] for label in labels], alpha=0.4, width=bar_width, label='Total Predictions')

        # Plot correct predictions
        bars_correct = ax.bar(labels, correct_preds, color=[colors[label] for label in labels], alpha=0.8, width=bar_width, label='Correct Predictions')

        # Add dotted lines for correct totals
        for i, (label, correct_total) in enumerate(zip(labels, correct_totals)):
            ax.plot([i - bar_width / 2, i + bar_width / 2], [correct_total, correct_total], linestyle='dotted', color='black', label='Correct Total' if i == 0 else "")
            ax.text(i, correct_total, f'{correct_total}', ha='center', va='bottom', fontsize=10, color='black')  # Add number to the dotted line

        ax.set_title(f"{title}\nTotal Accuracy: {total_accuracy:.1%}")
        ax.set_ylabel("Count")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")

        # Add number values above each bar
        for bar, total, correct in zip(bars_total, total_preds, correct_preds):
            ax.text(bar.get_x() + bar.get_width() / 2, total, f'{total}', ha='center', va='bottom', fontsize=10)
            ax.text(bar.get_x() + bar.get_width() / 2, correct, f'{correct}', ha='center', va='bottom', fontsize=10, color='black')
        
        # Add accuracy text inside the correct predictions bars
        for i, (bar, correct, correct_total) in enumerate(zip(bars_correct, correct_preds, correct_totals)):
            if correct > 0:  # Only show accuracy if there are correct predictions
                accuracy = correct / correct_total if correct_total > 0 else 0
                # Position the text at the top of the correct predictions bar
                text_y = correct * 0.9  # 90% of the bar height to keep it inside
                ax.text(bar.get_x() + bar.get_width() / 2, text_y, f'{accuracy:.1%}', 
                       ha='center', va='center', fontsize=10, color='white', weight='bold')

    # Add legend with gray color
    legend_handles = [
        mpatches.Patch(color='black', alpha=0.3, label='Total Predictions'),
        mpatches.Patch(color='black', alpha=0.7, label='Correct Predictions'),
        mpatches.Patch(color='black', linestyle='dotted', label='Correct Total')
    ]
    axes[0].legend(handles=legend_handles, loc='lower right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the heading
    plt.show()

def show_unsub_preds_per_error_type(data_dicts, titles):
    """
    Plots the distribution of error types within the unsubstantiated data rows and highlights the amount of correct predictions for each type.

    Each dictionary should have the following structure:
    {
        '1 - Irrelevant': {'total': 14, 'correct_class': 14, 'false_class': 0},
        '2 - Copy-Paste-Error': {'total': 3, 'correct_class': 0, 'false_class': 3},
        '3.1 - Misunderstanding-nosup': {'total': 1, 'correct_class': 0, 'false_class': 1},
        '3.2 - Misunderstanding-partsup': {'total': 5, 'correct_class': 1, 'false_class': 4},
        '4 - Unwanted': {'total': 8, 'correct_class': 6, 'false_class': 2}
    }
    """
    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    # Plot each dictionary
    for ax, data, title in zip(axes, data_dicts, titles):
        error_types = list(data.keys())
        total_counts = [data[error_type]["total"] for error_type in error_types]
        correct_counts = [data[error_type]["correct_class"] for error_type in error_types]

        # Bar positions
        x = np.arange(len(error_types))

        # Plot bars
        total_bars = ax.bar(x, total_counts, color='tab:blue', alpha=0.4, label='Total')
        correct_bars = ax.bar(x, correct_counts, color='tab:blue', alpha=1.0, label='Correct')

        # Add numbers above the bars
        for bar, total, correct in zip(total_bars, total_counts, correct_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{total}', ha='center', va='bottom', fontsize=10)
        for bar, correct in zip(correct_bars, correct_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{correct}', ha='center', va='bottom', fontsize=10)

        # Set title and labels
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(error_types, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

def show_accuracy_by_attribute_values(results_dict, attribute_name, y_min=70):
    """
    Display accuracy results as a bar plot with bars grouped by model and colors representing attribute values.
    
    Parameters:
    results_dict: Dictionary with structure {model_name: {attribute_value: {results...}}}
    attribute_name: The name of the attribute being analyzed
    y_min: Minimum value for y-axis (default: 60)
    """
    # Extract model names and attribute values
    model_names = list(results_dict.keys())
    
    # Get all unique attribute values from the first model (excluding 'Total')
    first_model = list(results_dict.keys())[0]
    attribute_values = [val for val in results_dict[first_model].keys() if val != 'Total']
    # Add 'Total' at the beginning for better visualization
    attribute_values = ['Total'] + attribute_values
    
    # Define colors for each attribute value (grey for Total, bright colors for others)
    bright_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E9', '#82E0AA', '#F8C471', '#F1948A', '#AED6F1', '#A3E4D7', '#D5A6BD',
        '#7FB3D3', '#76D7C4', '#F9E79F', '#D7BDE2', '#A9CCE3', '#A9DFBF', '#FCF3CF', '#FADBD8',
        '#D6EAF8', '#D1F2EB', '#FDEAA7', '#E8DAEF', '#D4E6F1', '#D0ECE7', '#FEF9E7', '#FDEDEC'
    ]
    colors = ['#808080']  # Grey for Total
    colors.extend(bright_colors[:len(attribute_values)-1])  # Bright colors for the rest
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate bar positions with adaptive gaps between bars in a group
    bar_width = 0.7 / len(attribute_values)  # Adjust width based on number of attribute values
    # Adaptive gap: smaller gaps for more attribute values
    if len(attribute_values) <= 3:
        gap = 0.05
    elif len(attribute_values) <= 5:
        gap = 0.03
    elif len(attribute_values) <= 7:
        gap = 0.02
    else:
        gap = 0.01
    
    x = np.arange(len(model_names))
    
    # Store total accuracies for difference calculation
    total_accuracies = {}
    for model_name in model_names:
        if 'Total' in results_dict[model_name] and 'accuracy' in results_dict[model_name]['Total']:
            total_accuracies[model_name] = results_dict[model_name]['Total']['accuracy'] * 100
        else:
            total_accuracies[model_name] = 0
    
    # Plot bars for each attribute value
    for i, attr_value in enumerate(attribute_values):
        accuracies = []
        for model_name in model_names:
            if attr_value in results_dict[model_name] and 'accuracy' in results_dict[model_name][attr_value]:
                accuracy = results_dict[model_name][attr_value]['accuracy']
                accuracies.append(accuracy * 100)  # Convert to percentage
            else:
                accuracies.append(0)  # Default to 0 if not found
        
        # Calculate offset for this attribute value's bars with gaps
        offset = (i - len(attribute_values)/2 + 0.5) * (bar_width + gap)
        
        # Create bars
        bars = ax.bar(x + offset, accuracies, bar_width, 
                     label=attr_value, color=colors[i], alpha=0.9)
        
        # Add value labels on top of bars
        for j, (bar, accuracy) in enumerate(zip(bars, accuracies)):
            if accuracy > 0:  # Only show label if there's a value
                model_name = model_names[j]
                
                # Add percentage label
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{accuracy:.1f}%', ha='center', va='bottom', fontsize=11)
                
                # Add G (total) count at the bottom of the bar
                if attr_value in results_dict[model_name] and 'G (Total)' in results_dict[model_name][attr_value]:
                    g_total = results_dict[model_name][attr_value]['G (Total)']
                    # Position text at the bottom of the bar (10% from bottom)
                    text_y = y_min + 1
                    ax.text(bar.get_x() + bar.get_width()/2, text_y,
                           f'{g_total}', ha='center', va='center', fontsize=10, 
                           color='black')
                
                # Add difference text for non-Total attributes
                if attr_value != 'Total' and model_name in total_accuracies:
                    difference = accuracy - total_accuracies[model_name]
                    if abs(difference) > 0.1:  # Only show if difference is meaningful
                        diff_color = 'green' if difference > 0 else 'red'
                        sign = '+' if difference > 0 else ''
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2.5,
                               f'{sign}{difference:.1f}%', ha='center', va='bottom', 
                               fontsize=10, color=diff_color, weight='bold')
    
    # Customize the plot
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Model Accuracy Comparison by "{attribute_name}"')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(title=f'Attribute Values', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis range with space above 100% but don't show 105% on axis
    ax.set_ylim(y_min, 105)
    ax.set_yticks(range(y_min, 101, 5))  # Only show ticks up to 100%
    
    plt.tight_layout()
    plt.show()