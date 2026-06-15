import numpy as np
import matplotlib.pyplot as plt
from mlquantify.representations import KDERepresentation

rng = np.random.default_rng(11)
X = np.concatenate([rng.normal(0.3, 0.12, 400),
                    rng.normal(0.65, 0.12, 300)]).reshape(-1, 1)
y = np.concatenate([np.zeros(400), np.ones(300)]).astype(int)
grid = np.linspace(-0.1, 1.1, 200).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8), sharey=True)
for ax, bw, tag in zip(axes, (0.02, 0.1, 0.4), ("too small", "good", "too large")):
    rep = KDERepresentation(bandwidth=bw).fit(X, y)
    ax.plot(grid[:, 0], np.exp(rep.class_representations_[0].score_samples(grid)),
            color="#4477aa", label="negative")
    ax.plot(grid[:, 0], np.exp(rep.class_representations_[1].score_samples(grid)),
            color="#cc6677", label="positive")
    ax.set_title(f"bandwidth = {bw} ({tag})", fontsize=9)
    ax.set_xlabel("feature")
axes[0].set_ylabel("density")
axes[0].legend(fontsize=8)
fig.tight_layout()