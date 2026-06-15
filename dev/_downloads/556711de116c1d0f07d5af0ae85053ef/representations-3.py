import numpy as np
import matplotlib.pyplot as plt
from mlquantify.representations import HistogramRepresentation

n = 12
edges = np.linspace(0, 1, n + 1)
centers = (edges[:-1] + edges[1:]) / 2
rng = np.random.default_rng(7)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3), sharey=True)
scenarios = [("well-separated classes", (2, 6), (6, 2)),
             ("overlapping classes",    (3, 3), (4, 3))]
for ax, (title, (an, bn), (ap, bp)) in zip(axes, scenarios):
    neg = rng.beta(an, bn, size=900)
    pos = rng.beta(ap, bp, size=600)
    scores = np.concatenate([neg, pos]).reshape(-1, 1)
    y = np.concatenate([np.zeros(900), np.ones(600)]).astype(int)
    rep = HistogramRepresentation(bins=(n,), range=(0, 1)).fit(scores, y)
    ax.bar(centers, rep.class_representations_[0], width=1 / n * 0.9,
           alpha=0.6, color="#4477aa", label="negative class")
    ax.bar(centers, rep.class_representations_[1], width=1 / n * 0.9,
           alpha=0.6, color="#cc6677", label="positive class")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("classifier score")
axes[0].set_ylabel("normalised mass")
axes[0].legend(fontsize=8)
fig.tight_layout()