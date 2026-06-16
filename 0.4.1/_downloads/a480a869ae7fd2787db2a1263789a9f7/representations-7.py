import numpy as np
import matplotlib.pyplot as plt
from mlquantify.representations import PredictionRepresentation

rng = np.random.default_rng(5)
proba = rng.dirichlet([3, 2, 2], size=500)   # 3-class posteriors
y = proba.argmax(1)
soft = np.asarray(PredictionRepresentation(method="soft").fit(proba, y).transform(proba))
hard = np.asarray(PredictionRepresentation(method="hard").fit(proba, y).transform(proba))

x = np.arange(3)
fig, ax = plt.subplots(figsize=(5.4, 3))
ax.bar(x - 0.19, soft, width=0.38, color="#2a7ab9",
       label="method='soft' (mean posterior)")
ax.bar(x + 0.19, hard, width=0.38, color="#b9542a",
       label="method='hard' (class frequency = CC)")
ax.set_xticks(x)
ax.set_xticklabels([f"class {i}" for i in x])
ax.set_ylabel("descriptor value")
ax.legend(fontsize=8)
fig.tight_layout()