import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from typing import List, Optional, Dict, Any, Union

plt.style.use("quantifyML/plots/style.mplstyle")

# Colors and markers
COLORS = [
    '#FFAB91', '#FFE082', '#A5D6A7', '#4DD0E1', '#FF6F61', '#FF8C94', '#D4A5A5',
    '#FF677D', '#B9FBC0', '#C2C2F0', '#E3F9A6', '#E2A8F7', '#F7B7A3', '#F7C6C7',
    '#8D9BFC', '#B4E6FF', '#FF8A65', '#FFC3A0', '#FFCCBC', '#F8BBD0', '#FF9AA2',
    '#FFB3B3', '#FFDDC1', '#FFE0B2', '#E2A8F7', '#F7C6C7', '#E57373', '#BA68C8',
    '#4FC3F7', '#FFB3B3', '#FF6F61'
]

MARKERS = ["o", "s", "^", "D", "p", "*", "+", "x", "H", "1", "2", "3", "4", "|", "_"]



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
    Plots a line plot based on the provided DataFrame and selected methods.
    """
    # Handle plot_params
    plot_params = plot_params or {}
    figsize = plot_params.pop('figsize', (10, 6))  # Default figsize if not provided

    # Filter data
    methods = methods or table_protocol['QUANTIFIER'].unique()
    table_protocol = table_protocol[table_protocol['QUANTIFIER'].isin(methods)]
    
    real = table_protocol["REAL_PREVS"].apply(lambda x: x[pos_alpha])
    table = table_protocol.drop(["PRED_PREVS", "REAL_PREVS"], axis=1).copy()
    table["ALPHA"] = real
    
    # Aggregate data
    if group_by:
        table = table.groupby(['QUANTIFIER', x])[y].agg(group_by).reset_index()

    # Create plot with custom figsize
    fig, ax = plt.subplots(figsize=figsize)
    for i, (method, marker) in enumerate(zip(methods, MARKERS[:len(methods)])):
        method_data = table[table['QUANTIFIER'] == method]
        y_data = real if y == "ALPHA" else method_data[y]
        ax.plot(method_data[x], y_data, color=COLORS[i % len(COLORS)], marker=marker, label=method, **plot_params)

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
