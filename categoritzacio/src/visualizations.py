import matplotlib.pyplot as plt
import numpy as np

def plot_label_distribution_with_missing(dataset):
    """
    Creates a bar plot of the label distribution in the dataset with a logarithmic y-axis.
    Counts and plots categories not in dataset.labels separately in another color.

    Args:
        dataset (CATEGORITZACIO): The dataset object.

    Returns:
        None
    """
    # Initialize label counts
    label_counts = np.zeros(len(dataset.labels))
    extra_categories = {}

    # Count label occurrences and track extra categories
    for categories in dataset.df['categories']:
        categories = categories.split(';')  # Split categories by the delimiter
        for category in categories:
            if category in dataset.labels:
                label_counts[dataset.labels.index(category)] += 1
            else:
                # Count occurrences of categories not in dataset.labels
                if category in extra_categories:
                    extra_categories[category] += 1
                else:
                    extra_categories[category] = 1

    # Prepare data for plotting
    extended_labels = dataset.labels + list(extra_categories.keys())  # Add extra categories
    extended_counts = np.append(label_counts, list(extra_categories.values()))  # Add their counts
    bar_colors = ['blue'] * len(dataset.labels) + ['orange'] * len(extra_categories)  # Different color for extra categories

    # Create a bar plot with a logarithmic y-axis
    plt.figure(figsize=(14, 8))
    plt.bar(extended_labels, extended_counts, color=bar_colors, alpha=0.75)
    plt.xlabel('Labels')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('Logarithmic Label Distribution with Extra Categories')
    plt.tight_layout()
    plt.savefig('label_distribution.png')

    # Print summary for clarity
    print(f"Total predefined categories: {len(dataset.labels)}")
    print(f"Categories with data: {np.count_nonzero(label_counts)}")
    print(f"Missing categories: {len(dataset.labels) - np.count_nonzero(label_counts)}")
    print(f"Extra categories (not in predefined labels): {len(extra_categories)}")
    if extra_categories:
        print("Extra categories and their counts:")
        for category, count in extra_categories.items():
            print(f"  {category}: {count}")
