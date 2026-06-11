import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.01, 0.99, 200)
pos = np.exp(-0.5 * ((x - 0.75) / 0.08) ** 2)
neg = np.exp(-0.5 * ((x - 0.25) / 0.08) ** 2)
mix = 0.6 * pos + 0.4 * neg
plt.plot(x, pos, label="positive KDE")
plt.plot(x, neg, label="negative KDE")
plt.plot(x, mix, linestyle="--", label="mixture")
plt.legend()