import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

plt.rcParams.update({
    'axes.facecolor': "#F8F8F8",
    'figure.facecolor': "#F8F8F8",
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'font.size': 12,
    'font.weight': 'light',
    'axes.labelsize': 14,
    'axes.labelweight': 'light',
    'axes.titlesize': 16,
    'axes.titleweight': 'normal',
    'boxplot.boxprops.linewidth': 0.3,
    'boxplot.whiskerprops.linewidth': 0.3,
    'boxplot.capprops.linewidth': 0.3,
    'boxplot.medianprops.linewidth': 0.6,
    'boxplot.flierprops.linewidth': 0.3,
    'boxplot.flierprops.markersize': 0.9,
    'boxplot.medianprops.color': 'black',
    'figure.subplot.bottom': 0.2,
    'axes.grid': True,
    'grid.color': 'black',
    'grid.alpha': 0.1,
    'grid.linewidth': 0.5,
    'grid.linestyle': '--'
})




COLORS = [
    '#FFAB91', '#FFE082', '#A5D6A7', '#4DD0E1', '#FF6F61', '#FF8C94', '#D4A5A5',
    '#FF677D', '#B9FBC0', '#C2C2F0', '#E3F9A6', '#E2A8F7', '#F7B7A3', '#F7C6C7',
    '#8D9BFC', '#B4E6FF', '#FF8A65', '#FFC3A0', '#FFCCBC', '#F8BBD0', '#FF9AA2',
    '#FFB3B3', '#FFDDC1', '#FFE0B2', '#E2A8F7', '#F7C6C7', '#E57373', '#BA68C8',
    '#4FC3F7', '#FFB3B3', '#FF6F61'
]


def class_distribution_plot(values: Union[List, np.ndarray],
                            labels: Union[List, np.ndarray],
                            bins: int = 30,
                            title: Optional[str] = None,
                            legend: bool = True,
                            save_path: Optional[str] = None,
                            plot_params: Optional[Dict[str, Any]] = None):
    """Plot overlaid histograms of class distributions.

    This function creates a plot with overlaid histograms, each representing the distribution
    of a different class or category. Custom colors, titles, legends, and other plot parameters 
    can be applied to enhance visualization.

    Args:
        values (Union[List, np.ndarray]): 
            A list of arrays or a single array containing values for specific classes or categories.
        labels (Union[List, np.ndarray]): 
            A list or an array of labels corresponding to each value set in `values`. 
            Must be the same length as `values`.
        bins (int, optional): 
            Number of bins to use in the histograms. Default is 30.
        title (Optional[str], optional): 
            Title of the plot. If not provided, no title will be displayed.
        legend (bool, optional): 
            Whether to display a legend. Default is True.
        save_path (Optional[str], optional): 
            File path to save the plot image. If not provided, the plot will not be saved.
        plot_params (Optional[Dict[str, Any]], optional): 
            Dictionary of custom plotting parameters to apply. Default is None.

    Raises:
        AssertionError: 
            If the number of labels does not match the number of value sets.

    """
    
    if isinstance(values, list):
        values = np.asarray(values)
    if isinstance(labels, list):
        labels = np.asarray(labels)
    
    
    # Apply custom plotting parameters if provided
    if plot_params:
        plt.rcParams.update(plot_params)

    # Ensure the number of labels matches the number of value sets
    assert len(values) == len(labels), "The number of value sets must match the number of labels."
    
    # Create the overlaid histogram
    for i, label in enumerate(np.unique(labels)):
        value_set = values[label == labels]
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

