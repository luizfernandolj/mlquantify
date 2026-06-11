import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
pos = np.sort(rng.normal(0.7, 0.12, 200))
neg = np.sort(rng.normal(0.3, 0.12, 200))
test = np.sort(np.concatenate([pos[:120], neg[:80]]))
y_pos = np.linspace(0, 1, len(pos))
y_neg = np.linspace(0, 1, len(neg))
y_test = np.linspace(0, 1, len(test))
plt.plot(pos, y_pos, label="positive CDF")
plt.plot(neg, y_neg, label="negative CDF")
plt.plot(test, y_test, linestyle="--", label="test CDF")
plt.legend()