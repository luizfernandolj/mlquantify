import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2)
pos = rng.normal(0.7, 0.1, 800)
neg = rng.normal(0.3, 0.1, 800)
mix = np.concatenate([pos[:600], neg[:400]])

bins = np.linspace(0, 1, 21)
plt.hist(pos, bins=bins, alpha=0.5, label="positive")
plt.hist(neg, bins=bins, alpha=0.5, label="negative")
plt.hist(mix, bins=bins, histtype="step", linewidth=2, label="test")
plt.xlim(0, 1)
plt.legend()