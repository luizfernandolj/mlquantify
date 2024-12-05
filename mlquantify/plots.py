import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas as pd
from typing import List, Optional, Dict, Any, Union




plt.rcParams.update({
    'lines.markersize': 6,
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

# Colors and markers
COLORS = [
    '#FFAB91', '#FFE082', '#A5D6A7', '#4DD0E1', '#FF6F61', '#FF8C94', '#D4A5A5',
    '#FF677D', '#B9FBC0', '#C2C2F0', '#E3F9A6', '#E2A8F7', '#F7B7A3', '#F7C6C7',
    '#8D9BFC', '#B4E6FF', '#FF8A65', '#FFC3A0', '#FFCCBC', '#F8BBD0', '#FF9AA2',
    '#FFB3B3', '#FFDDC1', '#FFE0B2', '#E2A8F7', '#F7C6C7', '#E57373', '#BA68C8',
    '#4FC3F7', '#FFB3B3', '#FF6F61'
]

MARKERS = ["o", "s", "^", "D", "p", "*", "+", "x", "H", "1", "2", "3", "4", "|", "_"]

def adjust_color_saturation(color: str, saturation_factor: float = 5) -> str:
    """
    Adjusts the saturation of a given color.
    
    Parameters
    ----------
    color : str
        Color in hexadecimal format.
    saturation_factor : float, optional
        Factor to adjust the saturation. Default is 5.
    
    Returns
    -------
    str
        Color in hexadecimal format with adjusted saturation.
    """
    # Convert color to HSV (Hue, Saturation, Value)
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
    
    # Adjust saturation
    s = min(1, s * saturation_factor)
    
    # Convert back to RGB and then to hex
    return mcolors.to_hex(mcolors.hsv_to_rgb((h, s, v)))



def protocol_boxplot(
    table_protocol: pd.DataFrame,
    x: str,
    y: str,
    methods: Optional[List[str]] = None,
    title: Optional[str] = None,
    legend: bool = True,
    save_path: Optional[str] = None,
    order: Optional[str] = None,
    plot_params: Optional[Dict[str, Any]] = None):
    """
    Plots a boxplot based on the provided DataFrame and selected methods.
    
    Parameters
    ----------
    table_protocol : pd.DataFrame
        DataFrame containing the protocol results.
    x : str
        Column name to use as the x-axis.
    y : str
        Column name to use as the y-axis.
    methods : List[str], optional
        List of quantifiers to plot. If not provided, all quantifiers will be plotted.
    title : str, optional
        Title of the plot. Default is None.
    legend : bool, optional
        Whether to display a legend. Default is True.
    save_path : str, optional
        File path to save the plot image. If not provided, the plot will not be saved.
    order : str, optional
        Order to plot the methods. If 'rank', methods will be ordered by median value.
    plot_params : Dict[str, Any], optional
        Dictionary of custom plotting parameters to apply. Default is None
    """
    # Handle plot_params
    plot_params = plot_params or {}
    figsize = plot_params.pop('figsize', (10, 6))  # Default figsize if not provided
    
    # Prepare data
    table = table_protocol.drop(["PRED_PREVS", "REAL_PREVS"], axis=1).copy()
    methods = methods or table['QUANTIFIER'].unique()
    table = table[table['QUANTIFIER'].isin(methods)]

    # Order methods by ranking if specified
    if order == 'rank':
        methods = table.groupby('QUANTIFIER')[y].median().sort_values().index.tolist()

    # Create plot with custom figsize
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    box = ax.boxplot([table[table['QUANTIFIER'] == method][y] for method in methods],
                     patch_artist=True, widths=0.8, labels=methods, **plot_params)

    # Apply colors
    for patch, color in zip(box['boxes'], COLORS[:len(methods)]):
        patch.set_facecolor(color)

    # Add legend
    if legend:
        handles = [mpatches.Patch(color=COLORS[i], label=method) for i, method in enumerate(methods)]
        ax.legend(handles=handles, title="Quantifiers", loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title_fontsize='11')

    # Customize plot
    ax.set_xticklabels(methods, rotation=45, fontstyle='italic')
    ax.set_xlabel(x.capitalize())
    ax.set_ylabel(f"{y.capitalize()}")
    if title:
        ax.set_title(title)

    # Adjust layout and save plot
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()







def protocol_lineplot(
    table_protocol: pd.DataFrame,
    methods: Union[List[str], str, None],
    x: str,
    y: str,
    title: Optional[str] = None,
    legend: bool = True,
    save_path: Optional[str] = None,
    group_by: str = "mean",
    pos_alpha: int = 1,
    plot_params: Optional[Dict[str, Any]] = None):
    """
    Plots a line plot based on the provided DataFrame of the protocol and selected methods.
    
    Parameters
    ----------
    table_protocol : pd.DataFrame
        DataFrame containing the protocol results.
    methods : Union[List[str], str, None]
        List of quantifiers to plot. If not provided, all quantifiers will be plotted.
    x : str
        Column name to use as the x-axis.
        - If 'ALPHA', the real prevalence of the positive class will be used.
        - You can also use any other column name, as long as the x has the same name.
    y : str
        Column name to use as the y-axis.
    title : str, optional
        Title of the plot. Default is None.
    legend : bool, optional
        Whether to display a legend. Default is True.
    save_path : str, optional
        File path to save the plot image. If not provided, the plot will not be saved.
    group_by : str, optional
        Column to group the data. Default is 'mean'.
    pos_alpha : int, optional
        Position of the positive class in the 'PREVS' column, this attribute only works for binary problems. Default is 1.
    plot_params : Dict[str, Any], optional
        Dictionary of custom plotting parameters to apply. Default is None.
    """
    # Handle plot_params
    plot_params = plot_params or {}
    figsize = plot_params.pop('figsize', (10, 6))  # Default figsize if not provided

    # Filter data
    methods = methods or table_protocol['QUANTIFIER'].unique()
    table_protocol = table_protocol[table_protocol['QUANTIFIER'].isin(methods)]
    
    if x == "ALPHA":
        real = table_protocol["REAL_PREVS"].apply(lambda x: x[pos_alpha])
        table = table_protocol.drop(["PRED_PREVS", "REAL_PREVS"], axis=1).copy()
        table["ALPHA"] = real
    else:
        table = table_protocol.drop(["PRED_PREVS", "REAL_PREVS"], axis=1).copy()
    
    # Aggregate data
    if group_by:
        table = table.groupby(['QUANTIFIER', x])[y].agg(group_by).reset_index()

    # Create plot with custom figsize
    fig, ax = plt.subplots(figsize=figsize)
    for i, (method, marker) in enumerate(zip(methods, MARKERS[:len(methods)+1])):
        method_data = table[table['QUANTIFIER'] == method]
        y_data = real if y == "ALPHA" else method_data[y]
        color = adjust_color_saturation(COLORS[i % len(COLORS)])  # Aumenta a saturação das cores
        ax.plot(method_data[x], 
                y_data, color=color, 
                marker=marker, 
                label=method,
                alpha=1.0, 
                **plot_params)

    # Add legend
    if legend:
        ax.legend(title="Quantifiers", loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title_fontsize='11')

    # Customize plot
    ax.set_xlabel(x.capitalize())
    ax.set_ylabel(y.capitalize())
    if title:
        ax.set_title(title)

    # Adjust layout and save plot
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

    Parameters
    ----------
    values : Union[List, np.ndarray]
        List or array of values to plot.
    labels : Union[List, np.ndarray]
        List or array of labels corresponding to the values.
    bins : int, optional
        Number of bins to use for the histogram. Default is 30.
    title : str, optional
        Title of the plot. Default is None.
    legend : bool, optional
        Whether to display a legend. Default is True.
    save_path : str, optional
        File path to save the plot image. If not provided, the plot will not be saved.
    plot_params : Dict[str, Any], optional
        Dictionary of custom plotting parameters to apply. Default is None.
    
    Raises
    ------
    AssertionError
        If the number of value sets does not match the number of labels.
    """
    
    # Ensure the number of labels matches the number of value sets
    assert len(values) == len(labels), "The number of value sets must match the number of labels."
    
    if isinstance(values, list):
        values = np.asarray(values)
    if isinstance(labels, list):
        labels = np.asarray(labels)
    
    
    # Apply custom plotting parameters if provided
    if plot_params:
        plt.rcParams.update(plot_params)

    if values.shape[1] > 1:
        num_plots = values.shape[1]  # Number of columns in `values`
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axs = axs.flatten()
    
    # Create the overlaid histogram
    for i, label in enumerate(np.unique(labels)):
        if values.shape[1] > 1:
            for j, lab in enumerate(np.unique(labels)):
                value_set = values[:, j][label == labels]
                axs[i].hist(value_set, bins=bins, color=COLORS[j % len(COLORS)], edgecolor='black', alpha=0.5, label=lab)
                axs[i].set_xlim([0, 1])  # Fix x-axis range between 0 and 1
        else:
            value_set = values[label == labels]
            plt.hist(value_set, bins=bins, color=COLORS[i % len(COLORS)], edgecolor='black', alpha=0.5, label=label)
            plt.xlim([0, 1])  # Fix x-axis range between 0 and 1

    if values.shape[1] > 1:
        for i in range(i + 1, len(axs)):
            fig.delaxes(axs[i])
    
    # Add title to the plot if provided
    if title:
        if values.shape[1] > 1:
            for i in range(values.shape[1]):
                axs[i].set_title(f'{title} for class {i+1}')
        else:
            plt.title(title)

    # Add legend to the plot if enabled
    if legend:
        if values.shape[1] > 1:
            for i in range(values.shape[1]):
                axs[i].legend(loc='upper right')
        else:
            plt.legend(loc='upper right')

    # Set axis labels
    if values.shape[1] > 1:
        for i in range(values.shape[1]):
            axs[i].set_xlabel('Values')
            axs[i].set_ylabel('Frequency')
    else:
        plt.xlabel('Values')
        plt.ylabel('Frequency')

    # Adjust layout to prevent overlapping
    plt.subplots_adjust(hspace=0.9, wspace=0.4)
    plt.tight_layout()

    # Save the figure if a path is specified
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.show()


