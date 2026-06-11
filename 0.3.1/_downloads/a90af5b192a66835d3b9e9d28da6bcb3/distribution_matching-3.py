import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 200)
gamma = 1.5
k_rbf = np.exp(-gamma * (x ** 2))
plt.plot(x, k_rbf, label="rbf kernel")
plt.axhline(0, color="0.8", linewidth=1)
plt.legend()