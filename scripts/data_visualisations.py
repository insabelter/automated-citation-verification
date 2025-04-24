import matplotlib.pyplot as plt
import pandas as pd

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

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', startangle=90, counterclock=False)
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