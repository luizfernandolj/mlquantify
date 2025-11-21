import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mlquantify.confidence import ConfidenceInterval, ConfidenceEllipseSimplex, ConfidenceEllipseCLR

np.random.seed(42)
X = np.random.dirichlet(np.ones(3), size=200)

def plot_pdf_with_region(data, region, class_label, region_name):
    kde = gaussian_kde(data)
    x = np.linspace(max(0, data.min()-0.05), min(1, data.max()+0.05), 500)
    y = kde(x)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, color='#7088F0', label='Estimated PDF')
    plt.fill_between(x, 0, y, alpha=0.3, color='#7088F0')

    if region_name == 'ConfidenceInterval':
        low, high = region.get_region()
        low_i, high_i = low[class_label-1], high[class_label-1]
        plt.scatter([low_i, high_i], [0, 0], color='#7212B7', s=100, label='Interval Limits')
        plt.axvline(low_i, color='#7212B7', linestyle='--', alpha=0.7)
        plt.axvline(high_i, color='#7212B7', linestyle='--', alpha=0.7)
        point_dentro = np.copy(region.get_point_estimate())
        point_fora = np.copy(region.get_point_estimate())
        point_fora[class_label-1] = high_i + 0.05
    else:
        center = region.get_point_estimate()
        deslocamento = 0.1
        point_dentro = np.copy(center)
        point_dentro[class_label-1] = np.clip(center[class_label-1] - deslocamento, 0, 1)
        point_fora = np.copy(center)
        point_fora[class_label-1] = np.clip(center[class_label-1] + 0.2, 0, 1)
        plt.scatter([center[class_label-1]], [0], color='darkred', s=100, label='Point Estimate (Mean)')
        plt.axvline(point_dentro[class_label-1], color='green', linestyle='--', linewidth=2, label='Inside Point')

    dentro_check = region.contains(point_dentro)
    fora_check = region.contains(point_fora)

    plt.scatter(point_dentro[class_label-1], 0, c='green' if dentro_check else 'orange', s=150, zorder=5, label='New Point INSIDE')
    plt.scatter(point_fora[class_label-1], 0, c='red' if not fora_check else 'blue', s=150, zorder=5, label='New Point OUTSIDE')

    plt.title(f'{region_name} - Class {class_label}', fontsize=16)
    plt.xlabel('Prevalence', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{region_name}_class_{class_label}.png')

region_classes = [
    ConfidenceInterval,
    ConfidenceEllipseSimplex,
    ConfidenceEllipseCLR,
]

for region_cls in region_classes:
    region = region_cls(X, confidence_level=0.95)
    plot_pdf_with_region(X[:, 0], region, class_label=1, region_name=region_cls.__name__)
