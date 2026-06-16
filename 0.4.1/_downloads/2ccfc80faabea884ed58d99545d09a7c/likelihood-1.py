import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

rng = np.random.default_rng(0)
X, y = make_classification(n_samples=4000, weights=[0.5, 0.5], random_state=0)
X_tr, X_te, y_tr, y_te = X[:2000], X[2000:], y[:2000], y[2000:]

# resample the test set to be 80% positive (prior probability shift)
pos, neg = np.where(y_te == 1)[0], np.where(y_te == 0)[0]
sel = np.concatenate([rng.choice(pos, 800), rng.choice(neg, 200)])
X_te, y_te = X_te[sel], y_te[sel]
true_prev = float(y_te.mean())

clf = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
post = clf.predict_proba(X_te)                 # P_L(y | x)
p_L = np.bincount(y_tr, minlength=2) / len(y_tr)

# --- the SLD / EMQ iteration ---
p, history, corrected = p_L.copy(), [p_L[1]], None
for _ in range(40):
    r = post * (p / p_L)                       # E-step
    r /= r.sum(axis=1, keepdims=True)
    p_new = r.mean(axis=0)                      # M-step
    history.append(p_new[1])
    if np.abs(p_new - p).max() < 1e-5:
        p, corrected = p_new, r[:, 1]
        break
    p = p_new
if corrected is None:
    corrected = (post * (p / p_L))
    corrected = (corrected / corrected.sum(axis=1, keepdims=True))[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].plot(range(len(history)), history, marker="o", ms=3, color="#2a7ab9")
axes[0].axhline(p_L[1], ls=":", color="gray",
                label=f"start = train prior ({p_L[1]:.2f})")
axes[0].axhline(true_prev, ls="--", color="#cc6677",
                label=f"true test prevalence ({true_prev:.2f})")
axes[0].set_xlabel("EM iteration")
axes[0].set_ylabel("estimated positive prevalence")
axes[0].set_title("estimate converges to the test prevalence", fontsize=10)
axes[0].legend(fontsize=8)

axes[1].hist(post[:, 1], bins=20, alpha=0.5, color="gray", label="raw posteriors")
axes[1].hist(corrected, bins=20, alpha=0.6, color="#2a9b5c", label="EM-corrected")
axes[1].set_xlabel("P(positive | x)")
axes[1].set_ylabel("count")
axes[1].set_title("posteriors re-weighted by EM", fontsize=10)
axes[1].legend(fontsize=8)
fig.tight_layout()