import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, Union

# Apply custom style from the .mplstyle file
plt.style.use("quantifyML/plots/style.mplstyle")

# Define colors
COLORS = [
    '#FFAB91', '#FFE082', '#A5D6A7', '#4DD0E1', '#FF6F61', '#FF8C94', '#D4A5A5',
    '#FF677D', '#B9FBC0', '#C2C2F0', '#E3F9A6', '#E2A8F7', '#F7B7A3', '#F7C6C7',
    '#8D9BFC', '#B4E6FF', '#FF8A65', '#FFC3A0', '#FFCCBC', '#F8BBD0', '#FF9AA2',
    '#FFB3B3', '#FFDDC1', '#FFE0B2', '#E2A8F7', '#F7C6C7', '#E57373', '#BA68C8',
    '#4FC3F7', '#FFB3B3', '#FF6F61'
]

def class_distribution_plot(values: List[np.ndarray],
                            labels: List[str],
                            bins: int=30,
                            title: Optional[str] = None,
                            legend: bool = True,
                            save_path: Optional[str] = None,
                            plot_params: Optional[Dict[str, Any]] = None):
    # Apply custom plotting parameters if provided
    if plot_params:
        plt.rcParams.update(plot_params)

    # Ensure the number of labels matches the number of value sets
    assert len(values) == len(labels), "The number of value sets must match the number of labels."

    # Create the overlaid histogram
    for i, (value_set, label) in enumerate(zip(values, labels)):
        plt.hist(value_set, bins=bins, color=COLORS[i % len(COLORS)], edgecolor='black', alpha=0.5, label=label)

    # Add title to the plot if provided
    if title:
        plt.title(title)

    # Add legend to the plot if enabled
    if legend:
        plt.legend(loc='upper right')

    # Set axis labels
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Save the figure if a path is specified
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.show()
