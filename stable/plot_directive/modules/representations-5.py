import numpy as np
import matplotlib.pyplot as plt
from mlquantify.representations import DistanceRepresentation

rng = np.random.default_rng(0)
X0 = rng.normal(-1.3, 0.8, (300, 6))
X1 = rng.normal(1.3, 0.8, (300, 6))
X = np.vstack([X0, X1])
y = np.r_[np.zeros(300), np.ones(300)].astype(int)

rep = DistanceRepresentation().fit(X, y)
d0 = np.asarray(rep.transform(X[y == 0]), float)   # sample drawn from class 0
d1 = np.asarray(rep.transform(X[y == 1]), float)   # sample drawn from class 1

x = np.arange(2)
fig, ax = plt.subplots(figsize=(5.6, 3))
ax.bar(x - 0.19, d0, width=0.38, color="#4477aa", label="sample from class 0")
ax.bar(x + 0.19, d1, width=0.38, color="#cc6677", label="sample from class 1")
ax.set_xticks(x)
ax.set_xticklabels(["mean dist to class 0", "mean dist to class 1"])
ax.set_ylabel("mean distance")
ax.legend(fontsize=8)
fig.tight_layout()