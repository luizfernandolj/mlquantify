import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Union, Dict, Any, Optional

def protocol_lineplot(
    table_protocol: pd.DataFrame,
    methods: Union[List[str], str, None],
    x: str,
    y: str,
    title: str = None,
    legend: bool = True,
    save_path: str = None,
    group_by: str = "mean",
    plot_params: Dict[str, Any] = None
):
    """
    Plots a line graph from the provided DataFrame.

    Parameters:
    - table_protocol (pd.DataFrame): DataFrame containing the data to plot.
    - methods (Union[List[str], str, None]): Methods to plot; if None or 'all', all methods are used.
    - x (str): Column name for x-axis.
    - y (str): Column name for y-axis.
    - title (str, optional): Title of the plot.
    - legend (bool, optional): Whether to display the legend.
    - save_path (str, optional): Path to save the plot.
    - group_by (str, optional): Aggregation method for y-axis values ('mean', 'sum', etc.).
    - plot_params (Dict[str, Any], optional): Additional plotting parameters.
    """
    # Determine methods to plot
    if methods is None or methods == "all":
        methods = table_protocol["QUANTIFIER"].unique()
    elif isinstance(methods, str):
        methods = [methods]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid()

    NUM_COLORS = len(methods)
    # Use colormap for more than 10 methods
    if NUM_COLORS > 10:
        cm = plt.get_cmap('tab20')
        ax.set_prop_cycle(color=[cm(i / NUM_COLORS) for i in range(NUM_COLORS)])
    else:
        # Default color cycle
        ax.set_prop_cycle(plt.cycler(color=plt.cm.tab10.colors))

    # Filter data for specified methods
    filtered_data = table_protocol[table_protocol["QUANTIFIER"].isin(methods)]

    # Apply additional plot parameters
    if plot_params:
        plt.rcParams.update(plot_params)

    # Group data and aggregate
    grouped_data = filtered_data.groupby([x, "QUANTIFIER"]).agg({y: group_by}).reset_index()

    # Plot lines for each method
    for method in methods:
        method_data = grouped_data[grouped_data["QUANTIFIER"] == method]
        ax.plot(method_data[x], method_data[y], label=method)
    
    # Set title and legend
    if title:
        ax.set_title(title)
    if legend:
        ax.legend()
    
    # Save plot if path is specified
    if save_path:
        plt.savefig(save_path)

    # Show plot
    plt.show()



def protocol_boxplot(
    table_protocol: pd.DataFrame,
    x: str,
    y: str,
    methods: Optional[List[str]] = None,
    title: str = None,
    legend: bool = True,
    save_path: str = None,
    plot_params: Dict[str, Any] = None
):
    """
    Plots a boxplot based on the provided DataFrame and selected methods.

    Parameters:
    - table_protocol (pd.DataFrame): DataFrame containing the data to plot.
    - methods (Optional[List[str]], optional): Methods to plot; if None, all methods are used.
    - x (str): Column name for x-axis (categorical).
    - y (str): Column name for y-axis (numeric).
    - title (str, optional): Title of the plot.
    - legend (bool, optional): Whether to display the legend (dummy for boxplots).
    - save_path (str, optional): Path to save the plot.
    - plot_params (Dict[str, Any], optional): Additional plotting parameters.
    """
    # Determine methods to plot
    if methods is None:
        methods = table_protocol["QUANTIFIER"].unique()
    elif not methods:
        methods = table_protocol["QUANTIFIER"].unique()

    # Filter data for specified methods
    filtered_data = table_protocol[table_protocol["QUANTIFIER"].isin(methods)]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid()

    # Apply additional plot parameters
    if plot_params:
        plt.rcParams.update(plot_params)

    # Define a colormap
    cm = plt.get_cmap('tab20')
    colors = [cm(i / len(methods)) for i in range(len(methods))]

    # Create the boxplot
    boxplot_data = [filtered_data[filtered_data["QUANTIFIER"] == method][y] for method in methods]
    boxprops = [dict(facecolor=colors[i], color=colors[i]) for i in range(len(methods))]
    ax.boxplot(boxplot_data, labels=methods, patch_artist=True,
               boxprops=boxprops)
    
    # Set title and labels
    if title:
        ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(f"Error: {y}")
    
    # Add legend if necessary (dummy legend for boxplot)
    if legend:
        handles = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(len(methods))]
        labels = methods
        ax.legend(handles, labels, title="Methods")

    # Save plot if path is specified
    if save_path:
        plt.savefig(save_path)

    # Show plot
    plt.show()
