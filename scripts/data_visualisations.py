import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.gridspec as gridspec

citation_functions = {
    "Use": ["Apply", "Extend"],
    "Compare": ["Criticize", "Contrast", "Confirm"],
    "Related": ["Definition/Proof", "Fundamentals", "Acknowledge"],
    "Background": ["Introduction/Bigger picture", "Unrelated/Unclear"]
}

def show_label_dist_comparison(label_dist, save_title=None):
    """
    Plot comparison of label distributions between original and re-annotated datasets.
    
    Parameters:
    label_dist: Dictionary with structure as shown in the example
    save_title: Optional title for saving the plot
    """
    # Define colors for each label type (consistent with other plot functions)
    colors = {
        "Fully Substantiated": "#2ca02c",      # Green (same as Accuracy)
        "Substantiated": "#2ca02c",            # Green (same as Accuracy)
        "Partially Substantiated": "#ffcc00",  # Orange-tinted yellow
        "Unsubstantiated": "#c62828"           # Slightly darker red for consistency
    }
    
    # Extract datasets and determine subplot widths based on number of bars
    datasets = list(label_dist.keys())
    dataset_items = [list(data.keys()) for data in label_dist.values()]
    num_bars = [len(items) for items in dataset_items]
    
    # Create subplots with widths proportional to number of bars (making both plots more balanced)
    fig = plt.figure(figsize=(12, 6))
    # Adjust ratio to better match bar widths: use 2.2:1.5 instead of 3:1.5
    width_ratios = [2.35, 1.5] if len(num_bars) >= 2 else [1, 1]
    gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    
    # Define fixed bar width (reduced for narrower bars)
    bar_width = 0.4
    
    # Store data for difference calculation
    all_data = []
    
    for i, (dataset_name, data) in enumerate(label_dist.items()):
        ax = axes[i]
        
        # Extract labels and counts
        labels = list(data.keys())
        counts = list(data.values())
        
        # Store data for later comparison
        all_data.append((labels, counts))
        
        # Map colors to labels
        bar_colors = [colors[label] for label in labels]
        
        # Create bar plot with fixed width and reduced spacing
        x_positions = np.arange(len(labels)) * 0.6  # Smaller spacing between bars
        bars = ax.bar(x_positions, counts, width=bar_width, color=bar_colors, alpha=0.8)
        
        # Set title and labels
        ax.set_title(f'{dataset_name}', fontsize=14)
        
        # For the right plot, add extra padding to align x-axis label with left plot
        if i == 1:
            ax.set_xlabel('Labels', fontsize=12, labelpad=29)
        else:
            ax.set_xlabel('Labels', fontsize=12)
            
        if i == 0:  # Only set y-label for the first subplot
            ax.set_ylabel('Count', fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
        
        # Add the count values above each bar
        max_height = max(counts)
        offset = max_height * 0.01  # 1% of the maximum height for consistent spacing
        
        for bar, count in zip(bars, counts):
            ax.annotate(str(count), (bar.get_x() + bar.get_width() / 2., bar.get_height() + offset), 
                       ha='center', va='bottom', fontsize=12)
        
        # For the right plot (re-annotated dataset), add difference labels within bars
        if i == 1 and len(all_data) >= 2:
            original_labels, original_counts = all_data[0]
            
            # Create mapping from original labels to counts
            original_map = {}
            original_map["Fully Substantiated"] = original_counts[original_labels.index("Fully Substantiated")] if "Fully Substantiated" in original_labels else 0
            original_map["Partially Substantiated"] = original_counts[original_labels.index("Partially Substantiated")] if "Partially Substantiated" in original_labels else 0
            original_map["Unsubstantiated"] = original_counts[original_labels.index("Unsubstantiated")] if "Unsubstantiated" in original_labels else 0
            
            # Calculate differences and add labels within bars
            for j, (bar, label, count) in enumerate(zip(bars, labels, counts)):
                if label == "Substantiated":
                    # Compare with Fully Substantiated only from original dataset
                    diff = count - original_map["Fully Substantiated"]
                elif label == "Unsubstantiated":
                    # Compare with original Unsubstantiated
                    diff = count - original_map["Unsubstantiated"]
                else:
                    diff = 0
                
                if diff != 0:
                    sign = "+" if diff > 0 else ""
                    # Position text at 95% of bar height to be very close to the top
                    text_y = count * 0.93
                    ax.text(bar.get_x() + bar.get_width() / 2., text_y, f'{sign}{diff}', 
                           ha='center', va='center', fontsize=11, color='black')
        
        # Set fixed y-axis to ensure bars aren't cut off
        ax.set_ylim(0, 160)
    
    # Adjust layout and show the plots
    # Share y-axis between subplots
    axes[1].sharey(axes[0])
    plt.tight_layout()
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_distribution_dict_comparison(data_dict, save_title=None, left_xlabel_pad=None, right_xlabel_pad=None, attribute_value_name_changes=None):
    """
    Plot comparison of two distributions from a dictionary.
    
    Parameters:
    data_dict: Dictionary with two keys, each containing a distribution dictionary
    save_title: Optional title for saving the plot
    left_xlabel_pad: Optional padding for the left plot's x-axis label
    right_xlabel_pad: Optional padding for the right plot's x-axis label
    attribute_value_name_changes: Optional dictionary for renaming attribute values
        {
            'attribute_name': {
                'original_value1': 'Display Name 1',
                'original_value2': 'Display Name 2'
            },
            ...
        }
    """
    # Extract the two distributions
    keys = list(data_dict.keys())
    if len(keys) != 2:
        raise ValueError("Dictionary must contain exactly 2 distributions")
    
    # Find the maximum value across both distributions for consistent y-axis
    all_values = []
    for distribution in data_dict.values():
        all_values.extend(distribution.values())
    max_value = max(all_values) if all_values else 1
    
    # Create subplots for side-by-side bar charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    for i, (title, distribution) in enumerate(data_dict.items()):
        ax = axes[i]
        
        # Apply name changes if provided for this attribute
        renamed_distribution = distribution.copy()
        if attribute_value_name_changes and title in attribute_value_name_changes:
            name_mapping = attribute_value_name_changes[title]
            renamed_distribution = {}
            for original_name, value in distribution.items():
                display_name = name_mapping.get(original_name, original_name)
                renamed_distribution[display_name] = value
        
        # Convert to pandas Series for easier handling
        series = pd.Series(renamed_distribution)
        
        # Create bar plot
        bars = series.plot(kind='bar', ax=ax)
        
        # Set title and labels
        ax.set_title(f'{title}', fontsize=14)
        
        # Apply custom padding for x-axis labels if specified
        if i == 0 and left_xlabel_pad is not None:
            ax.set_xlabel('Annotation Attribute Value', fontsize=14, labelpad=left_xlabel_pad)
        elif i == 1 and right_xlabel_pad is not None:
            ax.set_xlabel('Annotation Attribute Value', fontsize=14, labelpad=right_xlabel_pad)
        else:
            ax.set_xlabel('Annotation Attribute Value', fontsize=14)
            
        if i == 0:  # Only set y-label for the first subplot
            ax.set_ylabel('Count', fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        ax.tick_params(axis='y', labelsize=11)  # Set y-axis tick label font size
        
        # Add the count values above each bar
        offset = max_value * 0.01  # 1% of the maximum height for consistent spacing
        
        for bar in bars.patches:
            height = bar.get_height()
            if height > 0:  # Only add text for non-zero bars
                ax.annotate(str(int(height)), 
                           (bar.get_x() + bar.get_width() / 2., height + offset), 
                           ha='center', va='bottom', fontsize=12)
        
        # Set consistent y-axis for both plots
        ax.set_ylim(0, max_value * 1.1)  # 10% extra space at the top
    
    # Adjust layout and show the plots
    plt.tight_layout()
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_distribution(df, column_name, include_nan=True, sorting=None, save_title=None):
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
    if not save_title:
        plt.title('Distribution of column: ' + column_name)
    plt.xlabel(column_name)
    plt.ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Add the total number above each bar with consistent spacing
    max_height = max([p.get_height() for p in ax.patches])
    offset = max_height * 0.01  # 1% of the maximum height for consistent spacing
    
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height() + offset), 
                    ha='center', va='bottom')
    
    # Adjust y-axis to provide space for labels
    ax.set_ylim(0, max_height * 1.1)  # 10% extra space at the top

    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_distribution_pie(df, column_name, include_nan=True, sorting=None, save_title=None):
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
    if not save_title:
        plt.title('Distribution of column: ' + column_name)
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_distribution_comparison(df1, df2, column_name, df1_name="Full Data", df2_name="False Predicted Data", include_nan=True, save_title=None):
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
    if not save_title:
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
    if not save_title:
        axes[1].set_title(f'Distribution ({column_name}): {df2_name}')
    axes[1].set_xlabel(column_name)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    # Add the total number above each bar for the second dataframe
    for p in ax2.patches:
        ax2.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height() + 1), 
                     ha='center', va='center')

    # Adjust layout and show the plots
    plt.tight_layout()
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_distribution_comparison_pie(df1, df2, column_name, df1_name="Full Data", df2_name="False Predicted Data", include_nan=True, label_threshold_percent=5, save_title=None):
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
    if not save_title:
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
    if not save_title:
        axes[1].set_title(f'Distribution ({column_name}): {df2_name}')

    # Adjust layout and show the plots
    plt.tight_layout()
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_retracted_distribution_pie(df, save_title=None):
    # Create a new column to categorize the rows
    df['Retracted Status'] = df.apply(lambda row: 'Reference and Citing' if row['Citing Article Retracted'] == 'Yes' and row['Reference Article Retracted'] == 'Yes' else 
                                    ('Citing Only' if row['Citing Article Retracted'] == 'Yes' else 
                                    ('Reference Only' if row['Reference Article Retracted'] == 'Yes' else 'None Retracted')), axis=1)

    # Count the occurrences of each category
    retracted_counts = df['Retracted Status'].value_counts()

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(retracted_counts, labels=retracted_counts.index, autopct='%1.1f%%', startangle=90, counterclock=False)
    if not save_title:
        plt.title('Distribution of Retracted Articles')
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_source_distribution_citing_article_retracted(df, save_title=None):
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

    if not save_title:
        plt.title('Distribution of Sources with Citing Article Retracted Percentage')
    plt.xlabel('Source')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_citation_function_main_distribution_pie(df, show_totals=False, save_title=None):
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
    if not save_title:
        plt.title('Distribution of column: ' + "Citation Function: Main")
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
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
    
def show_citation_function_sub_distribution_pie(df, show_totals=False, save_title=None):
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
    if not save_title:
        plt.title('Distribution of column: ' + "Citation Function: Sub")
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_preds_vs_correct_preds_vs_total(data_dicts, titles, title="Comparison of Accuracies, Predictions, Correct Predictions and Correct Totals per Label across Datasets", labels=['unsubstantiate', 'fully substantiate'], smaller_figures=False, save_title=None):
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
        if not save_title:
            fig.suptitle(title, fontsize=12)
    else:
        fig, axes = plt.subplots(1, len(data_dicts), figsize=(18, 6), sharey=True)
        if not save_title:
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

        if not save_title:
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
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_metrics_per_label(model_results, title="Model Performance Metrics by Label", save_title=None):
    # Define colors for each metric
    colors = {
        'Accuracy': '#87ceeb',        # Light Blue
        'Balanced Accuracy': '#1f77b4', # Blue
        'Precision': '#d62728',       # Red
        'Recall': '#2ca02c',          # Green
        'F1 Score': '#ff7f0e'         # Orange
    }
    
    # Create subplots - maximum 2 models per row
    num_models = len(model_results)
    num_cols = min(2, num_models)  # Maximum 2 columns
    num_rows = (num_models + 1) // 2  # Calculate rows needed
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 8 * num_rows), sharey=True)
    
    # Handle different cases for axes array
    if num_models == 1:
        axes = [axes]
    elif num_rows == 1:
        # Single row with multiple columns
        axes = axes if num_cols > 1 else [axes]
    else:
        # Multiple rows - flatten the array for easier indexing
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=18)
    
    # Define bar positions and width
    labels = ['Total', 'Substantiated', 'Unsubstantiated']
    total_metrics = ['Accuracy', 'Balanced Accuracy']
    label_metrics = ['Precision', 'Recall', 'F1 Score']
    bar_width = 0.35
    
    # Plot for each model
    for idx, (model_name, results) in enumerate(model_results.items()):
        ax = axes[idx]
        
        # Set up x positions for bars with reduced spacing between groups
        x = np.array([0, 1.4, 2.8])  # Further reduced spacing between groups
        
        # Plot Total metrics (Accuracy and Balanced Accuracy)
        total_values = []
        for metric in total_metrics:
            if metric == 'Accuracy':
                value = results.get('Accuracy', 0) * 100
            elif metric == 'Balanced Accuracy':
                value = results.get('Balanced Accuracy', 0) * 100
            else:
                value = 0
            total_values.append(value)
        
        # Plot Total bars
        for i, metric in enumerate(total_metrics):
            offset = (i - len(total_metrics)/2 + 0.5) * bar_width
            bars = ax.bar(x[0] + offset, total_values[i], bar_width, 
                         label=metric, color=colors[metric], alpha=0.8)
            
            # Add value labels on top of bars
            if total_values[i] > 0:
                ax.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 1,
                       f'{total_values[i]:.1f}%', ha='center', va='bottom', fontsize=12)
        
        # Extract metrics for both label types
        substantiated_metrics = []
        unsubstantiated_metrics = []
        
        for metric in label_metrics:
            # Get substantiated metrics
            if metric in results.get('Substantiated', {}):
                substantiated_metrics.append(results['Substantiated'][metric] * 100)
            else:
                substantiated_metrics.append(0)
            
            # Get unsubstantiated metrics
            if metric in results.get('Unsubstantiated', {}):
                unsubstantiated_metrics.append(results['Unsubstantiated'][metric] * 100)
            else:
                unsubstantiated_metrics.append(0)
        
        # Plot bars for label-specific metrics
        for i, metric in enumerate(label_metrics):
            offset = (i - len(label_metrics)/2 + 0.5) * bar_width
            
            # Substantiated bars
            bars_sub = ax.bar(x[1] + offset, substantiated_metrics[i], bar_width, 
                             label=metric if idx == 0 else "", color=colors[metric], alpha=0.8)
            
            # Unsubstantiated bars
            bars_unsub = ax.bar(x[2] + offset, unsubstantiated_metrics[i], bar_width, 
                               color=colors[metric], alpha=0.8)
            
            # Add value labels on top of bars
            if substantiated_metrics[i] > 0:
                ax.text(bars_sub[0].get_x() + bars_sub[0].get_width()/2, bars_sub[0].get_height() + 1,
                       f'{substantiated_metrics[i]:.1f}%', ha='center', va='bottom', fontsize=12)
            
            if unsubstantiated_metrics[i] > 0:
                ax.text(bars_unsub[0].get_x() + bars_unsub[0].get_width()/2, bars_unsub[0].get_height() + 1,
                       f'{unsubstantiated_metrics[i]:.1f}%', ha='center', va='bottom', fontsize=12)
        
        # Customize subplot
        ax.set_title(f'{model_name}', fontsize=16)
        ax.set_ylabel('Percentage (%)', fontsize=16)
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Increase tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Add legend only to the first subplot
        if idx == 0:
            ax.legend(title='Metrics', loc='lower right', fontsize=14, title_fontsize=16)
    
    # Hide any unused subplots if we have an odd number of models
    if num_models % 2 == 1 and num_models > 1:
        axes[num_models].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the heading
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_unsub_preds_per_error_type(data_dicts, titles, save_title=None):
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
        if not save_title:
            ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(error_types, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def get_division_text_for_metric(metric_name, attr_results):
    """
    Calculate the division text for a given metric based on the attribute results.
    
    Parameters:
    metric_name: The name of the metric (e.g., 'Accuracy', 'Substantiated Precision')
    attr_results: The results dictionary for a specific attribute value
    
    Returns:
    String representing the division calculation or empty string if not applicable
    """
    division_text = ""
    
    if metric_name == 'Accuracy':
        # Accuracy = (True Classifications for both labels) / Total
        sub_true = attr_results.get('Substantiated', {}).get('True Classifications', 0)
        unsub_true = attr_results.get('Unsubstantiated', {}).get('True Classifications', 0)
        total = attr_results.get('Total', 0)
        if total > 0:
            division_text = f"{sub_true + unsub_true} / {total}"
    
    elif metric_name.startswith('Substantiated'):
        sub_results = attr_results.get('Substantiated', {})
        if 'Precision' in metric_name:
            # Precision = True Classifications / (True Classifications + False Classifications by other label)
            true_class = sub_results.get('True Classifications', 0)
            unsub_false = attr_results.get('Unsubstantiated', {}).get('False Classifications', 0)
            total_preds = true_class + unsub_false
            if total_preds > 0:
                division_text = f"{true_class} / {total_preds}"
        elif 'Recall' in metric_name:
            # Recall = True Classifications / Label Total
            true_class = sub_results.get('True Classifications', 0)
            label_total = sub_results.get('Label Total', 0)
            if label_total > 0:
                division_text = f"{true_class} / {label_total}"
    
    elif metric_name.startswith('Unsubstantiated'):
        unsub_results = attr_results.get('Unsubstantiated', {})
        if 'Precision' in metric_name:
            # Precision = True Classifications / (True Classifications + False Classifications by other label)
            true_class = unsub_results.get('True Classifications', 0)
            sub_false = attr_results.get('Substantiated', {}).get('False Classifications', 0)
            total_preds = true_class + sub_false
            if total_preds > 0:
                division_text = f"{true_class} / {total_preds}"
        elif 'Recall' in metric_name:
            # Recall = True Classifications / Label Total
            true_class = unsub_results.get('True Classifications', 0)
            label_total = unsub_results.get('Label Total', 0)
            if label_total > 0:
                division_text = f"{true_class} / {label_total}"
    
    return division_text

def show_metrics_by_attribute_values(results_dict, attribute_name, model_name, save_title=None):
    """
    Display one plot per metric (Accuracy, Balanced Accuracy, Unsubstantiated Precision/Recall/F1, Substantiated Precision/Recall/F1).
    Each plot shows the values for that metric across Total and all attribute values.
    
    Parameters:
    results_dict: Dictionary with structure {model_name: {attribute_value: {results...}}}
    attribute_name: The name of the attribute being analyzed
    model_name: The specific model to analyze
    """
    # Define colors for each metric (matching previous usage)
    colors = {
        'Accuracy': '#87ceeb',                    # Light Blue
        'Balanced Accuracy': '#1f77b4',           # Blue
        'Unsubstantiated: Precision': '#d62728',   # Red
        'Unsubstantiated: Recall': '#2ca02c',      # Green
        'Unsubstantiated: F1 Score': '#ff7f0e',    # Orange
        'Substantiated: Precision': '#d62728',     # Red
        'Substantiated: Recall': '#2ca02c',        # Green
        'Substantiated: F1 Score': '#ff7f0e'       # Orange
    }
    
    # Get model results
    model_results = results_dict[model_name]
    
    # Extract attribute values and ensure Total is first
    attribute_values = [val for val in model_results.keys() if val != 'Total']
    attribute_values = ['Total'] + attribute_values
    
    # Define the metrics to plot with their positions
    metrics_to_plot = [
        ('Accuracy', 0, 0),
        ('Balanced Accuracy', 0, 1), 
        ('Unsubstantiated: Precision', 1, 0),
        ('Unsubstantiated: Recall', 1, 1),
        ('Unsubstantiated: F1 Score', 1, 2),
        ('Substantiated: Precision', 2, 0),
        ('Substantiated: Recall', 2, 1),
        ('Substantiated: F1 Score', 2, 2)
    ]
    
    # Calculate grid dimensions (3 columns, 3 rows)
    num_cols = 3
    num_rows = 3
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows), sharey=True)
    
    # Handle different cases for axes array
    if num_rows == 1:
        if num_cols == 1:
            axes = [axes]
        else:
            axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    if not save_title:
        fig.suptitle(f'{model_name} - Metrics by "{attribute_name}"', fontsize=16)
    
    # Plot each metric
    for metric_name, row, col in metrics_to_plot:
        ax = axes[row, col]
        
        # Extract values for this metric across all attribute values
        metric_values = []
        
        for attr_value in attribute_values:
            attr_results = model_results.get(attr_value, {})
            
            if metric_name == 'Accuracy':
                value = attr_results.get('Accuracy', 0) * 100
            elif metric_name == 'Balanced Accuracy':
                value = attr_results.get('Balanced Accuracy', 0) * 100
            elif metric_name.startswith('Unsubstantiated'):
                unsub_results = attr_results.get('Unsubstantiated', {})
                if 'Precision' in metric_name:
                    value = unsub_results.get('Precision', 0) * 100
                elif 'Recall' in metric_name:
                    value = unsub_results.get('Recall', 0) * 100
                elif 'F1 Score' in metric_name:
                    value = unsub_results.get('F1 Score', 0) * 100
                else:
                    value = 0
            elif metric_name.startswith('Substantiated'):
                sub_results = attr_results.get('Substantiated', {})
                if 'Precision' in metric_name:
                    value = sub_results.get('Precision', 0) * 100
                elif 'Recall' in metric_name:
                    value = sub_results.get('Recall', 0) * 100
                elif 'F1 Score' in metric_name:
                    value = sub_results.get('F1 Score', 0) * 100
                else:
                    value = 0
            else:
                value = 0
                
            metric_values.append(value)
        
        # Get Total value for difference calculation
        total_value = metric_values[0] if len(metric_values) > 0 else 0
        
        # Create bars
        x = np.arange(len(attribute_values))
        bars = ax.bar(x, metric_values, color=colors[metric_name], alpha=0.8)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, metric_values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Add difference text inside bars (skip Total bar)
        for i, (bar, value, attr_value) in enumerate(zip(bars, metric_values, attribute_values)):
            if attr_value != 'Total' and value > 0:
                diff = value - total_value
                if abs(diff) > 0.1:  # Only show if difference is meaningful
                    sign = '+' if diff > 0 else ''
                    text_y = value - 6  # Position with consistent 6% spacing from top of bar
                    ax.text(bar.get_x() + bar.get_width()/2, text_y,
                           f'{sign}{diff:.1f}%', ha='center', va='center', 
                           fontsize=10, color='black')
        
        # Add division text at bottom of bars (skip F1 Score and Balanced Accuracy)
        if 'F1 Score' not in metric_name and 'Balanced Accuracy' not in metric_name:
            for i, (bar, attr_value) in enumerate(zip(bars, attribute_values)):
                attr_results = model_results.get(attr_value, {})
                division_text = get_division_text_for_metric(metric_name, attr_results)
                
                # Add division text at bottom of bar
                if division_text:
                    ax.text(bar.get_x() + bar.get_width()/2, 2,
                           division_text, ha='center', va='bottom', 
                           fontsize=8, color='black')
        
        # Customize subplot
        ax.set_title(f'{metric_name}', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_ylim(0, 110)  # Extra space for labels
        ax.set_xticks(x)
        ax.set_xticklabels(attribute_values, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide the unused subplot (first row, last column)
    axes[0, 2].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the heading
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_best_models_comparison(best_model_configs, model_names=None, title="Best Model Configurations: Balanced Accuracy vs Unsubstantiated F1 Score", save_title=None):
    """
    Create a grouped bar chart comparing balanced accuracy and unsubstantiated F1 score for each model.
    
    Parameters:
    best_model_configs: Dictionary with structure as shown in the example
    model_names: Optional dictionary mapping model keys to display names
    title: Title for the plot
    """
    # Define colors matching the existing color scheme
    colors = {
        'Balanced Accuracy': '#1f77b4',  # Blue (matching existing scheme)
        'Unsubstantiated F1': '#ff7f0e'  # Orange (matching existing scheme)
    }
    
    # Extract data
    models = list(best_model_configs.keys())
    balanced_accuracies = [best_model_configs[model]['balanced_accuracy'] * 100 for model in models]
    unsub_f1_scores = [best_model_configs[model]['results']['Unsubstantiated']['F1 Score'] * 100 for model in models]
    
    # Get display names for models
    if model_names:
        display_names = [model_names.get(model, model) for model in models]
    else:
        display_names = models
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars and positions
    bar_width = 0.35
    x = np.arange(len(models))
    
    # Create bars
    bars1 = ax.bar(x - bar_width/2, balanced_accuracies, bar_width, 
                   label='Balanced Accuracy', color=colors['Balanced Accuracy'], alpha=0.8)
    bars2 = ax.bar(x + bar_width/2, unsub_f1_scores, bar_width,
                   label='Unsubstantiated F1 Score', color=colors['Unsubstantiated F1'], alpha=0.8)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
    
    # Customize the plot
    ax.set_xlabel('Models', fontsize=16)
    ax.set_ylabel('Percentage (%)', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=14)
    ax.legend(fontsize=14, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Adjust layout and show
    plt.tight_layout()
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()

def show_metrics_by_annotation_attributes(annotation_results_dict, save_title=None, attribute_names=None, attribute_value_orders=None):
    """
    Display metrics for each annotation attribute in a 2-column grid layout.
    Shows Balanced Accuracy and Unsubstantiated F1 Score for Total and each attribute value.
    
    Parameters:
    annotation_results_dict: Dictionary with structure:
        {
            'attribute_name': {
                'Total': {'Balanced Accuracy': ..., 'Unsubstantiated': {'F1 Score': ...}},
                'attribute_value1': {'Balanced Accuracy': ..., 'Unsubstantiated': {'F1 Score': ...}},
                ...
            }
        }
    save_title: Optional title for saving the plot
    attribute_names: Optional dictionary mapping original attribute names to display names
        {
            'original_attribute_name': 'Display Attribute Name',
            ...
        }
    attribute_value_orders: Optional dictionary defining the order and optionally renaming of attribute values for specific attributes
        Can contain either lists (order only) or dictionaries (order + renaming):
        {
            'attribute_name': ['value1', 'value2', 'value3'],  # List: order only
            'other_attribute': {                                # Dict: order + renaming
                'original_value1': 'Display Name 1',
                'original_value2': 'Display Name 2'
            },
            ...
        }
        Total is always plotted first regardless of this parameter.
    """
    # Define colors matching the existing color scheme
    colors = {
        'Balanced Accuracy': '#1f77b4',  # Blue (matching existing scheme)
        'Unsubstantiated F1': '#ff7f0e'  # Orange (matching existing scheme)
    }
    
    # Get attribute names - use order from attribute_names dict if provided
    if attribute_names:
        # Use the order from attribute_names dict, but only include attributes that exist in the data
        attributes = [attr for attr in attribute_names.keys() if attr in annotation_results_dict]
        # Add any remaining attributes from the data that aren't in attribute_names
        remaining_attrs = [attr for attr in annotation_results_dict.keys() if attr not in attribute_names]
        attributes.extend(remaining_attrs)
    else:
        attributes = list(annotation_results_dict.keys())
    
    num_attributes = len(attributes)
    
    # Calculate grid dimensions (2 columns)
    num_cols = 2
    num_rows = (num_attributes + 1) // 2  # Round up
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6 * num_rows))
    
    # Handle different cases for axes array
    if num_rows == 1 and num_cols == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    if not save_title:
        fig.suptitle('Model Performance by Annotation Attributes', fontsize=18)
    
    # Plot each attribute
    for idx, attribute_name in enumerate(attributes):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        
        # Get display name for attribute
        display_name = attribute_names.get(attribute_name, attribute_name) if attribute_names else attribute_name
        
        # Get attribute results
        attr_data = annotation_results_dict[attribute_name]
        
        # Extract attribute values and ensure Total is first
        attr_values_without_total = [val for val in attr_data.keys() if val != 'Total']
        
        # Apply custom ordering and renaming if provided for this attribute
        attr_value_display_mapping = {}  # Maps original values to display values
        if attribute_value_orders and attribute_name in attribute_value_orders:
            order_info = attribute_value_orders[attribute_name]
            
            if isinstance(order_info, list):
                # List format: order only, no renaming
                ordered_values = order_info
                # Use the custom order, but only include values that exist in the data
                attr_values_ordered = [val for val in ordered_values if val in attr_values_without_total]
                # Add any remaining values from the data that aren't in the custom order
                remaining_values = [val for val in attr_values_without_total if val not in ordered_values]
                attr_values_without_total = attr_values_ordered + remaining_values
                
            elif isinstance(order_info, dict):
                # Dict format: order + renaming
                ordered_values = list(order_info.keys())
                # Store the display name mapping
                attr_value_display_mapping = order_info
                # Use the custom order, but only include values that exist in the data
                attr_values_ordered = [val for val in ordered_values if val in attr_values_without_total]
                # Add any remaining values from the data that aren't in the custom order
                remaining_values = [val for val in attr_values_without_total if val not in ordered_values]
                attr_values_without_total = attr_values_ordered + remaining_values
        
        attr_values = ['Total'] + attr_values_without_total
        
        # Extract metrics for each attribute value
        balanced_accuracies = []
        unsub_f1_scores = []
        formatted_attr_values = []
        
        for attr_value in attr_values:
            value_data = attr_data[attr_value]
            
            # Extract Balanced Accuracy (convert to percentage)
            balanced_acc = value_data.get('Balanced Accuracy', 0) * 100
            balanced_accuracies.append(balanced_acc)
            
            # Extract Unsubstantiated F1 Score (convert to percentage)
            unsub_data = value_data.get('Unsubstantiated', {})
            unsub_f1 = unsub_data.get('F1 Score', 0) * 100
            unsub_f1_scores.append(unsub_f1)
            
            # Get total count for this attribute value and format the label
            total_count = value_data.get('Total', 0)
            
            # Use display name if mapping is provided, otherwise use original name
            display_value = attr_value_display_mapping.get(attr_value, attr_value)
            formatted_label = f"{display_value}\n({total_count})"
            formatted_attr_values.append(formatted_label)
        
        # Set width of bars and positions
        bar_width = 0.35
        x = np.arange(len(attr_values))
        
        # Create bars
        bars1 = ax.bar(x - bar_width/2, balanced_accuracies, bar_width,
                       label='Balanced Accuracy' if idx == 0 else "", 
                       color=colors['Balanced Accuracy'], alpha=0.8)
        bars2 = ax.bar(x + bar_width/2, unsub_f1_scores, bar_width,
                       label='Unsubstantiated F1 Score' if idx == 0 else "", 
                       color=colors['Unsubstantiated F1'], alpha=0.8)
        
        # Add value labels on top of bars
        for bar, value in zip(bars1, balanced_accuracies):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        for bar, value in zip(bars2, unsub_f1_scores):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Add difference text inside bars (skip Total bar)
        total_balanced_acc = balanced_accuracies[0] if balanced_accuracies else 0
        total_unsub_f1 = unsub_f1_scores[0] if unsub_f1_scores else 0
        
        for i, (bar1, bar2, balanced_acc, unsub_f1, attr_value) in enumerate(zip(bars1, bars2, balanced_accuracies, unsub_f1_scores, attr_values)):
            if attr_value != 'Total' and balanced_acc > 0:
                # Difference for Balanced Accuracy
                diff_acc = balanced_acc - total_balanced_acc
                if abs(diff_acc) > 0.1:
                    sign = '+' if diff_acc > 0 else ''
                    text_y = balanced_acc - 3  # Position near top of bar
                    ax.text(bar1.get_x() + bar1.get_width()/2, text_y,
                           f'{sign}{diff_acc:.1f}%', ha='center', va='center', 
                           fontsize=9, color='black')
            
            if attr_value != 'Total' and unsub_f1 > 0:
                # Difference for Unsubstantiated F1
                diff_f1 = unsub_f1 - total_unsub_f1
                if abs(diff_f1) > 0.1:
                    sign = '+' if diff_f1 > 0 else ''
                    text_y = unsub_f1 - 3  # Position near top of bar
                    ax.text(bar2.get_x() + bar2.get_width()/2, text_y,
                           f'{sign}{diff_f1:.1f}%', ha='center', va='center', 
                           fontsize=9, color='black')
        
        # Customize subplot
        ax.set_title(f'{display_name}', fontsize=14)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(formatted_attr_values, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend only to the first subplot
        if idx == 0:
            ax.legend(fontsize=12, loc='lower right')
    
    # Hide any unused subplots
    if num_attributes % 2 == 1 and num_attributes > 1:
        if num_rows > 1:
            axes[num_rows - 1, num_cols - 1].set_visible(False)
        else:
            axes[num_cols - 1].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the heading
    if save_title:
        plt.savefig(f"plots/{save_title}.pdf", bbox_inches="tight")
    plt.show()