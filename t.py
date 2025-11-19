from mlquantify.likelihood import EMQ
from mlquantify.utils import get_prev_from_labels, get_indexes_with_prevalence
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurar estilo seaborn
sns.set_style("whitegrid")
sns.set_context("notebook")

X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, n_redundant=10, n_repeated=2, n_classes=2, n_clusters_per_class=1, flip_y=0.1, class_sep=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

indexes = get_indexes_with_prevalence(y_test, prevalence=[0.9, 0.1], sample_size=600)
X_test = X_test[indexes]
y_test = y_test[indexes]

clf = RandomForestClassifier(n_estimators=100, random_state=42)

priors = get_prev_from_labels(y_train, format="array")
posteriors = clf.fit(X_train, y_train).predict_proba(X_test)
test_prev = get_prev_from_labels(y_test, format="array")[1]

q = EMQ(clf, calib_function="bcts")

all_priors = []
all_priors.append(priors[1])

for i in range(1, 10):
    p, pos = q.EM(priors=priors, posteriors=posteriors, max_iter=i)
    all_priors.append(p[1])
   
print(all_priors)

# Criar figura com estilo da imagem
fig, ax = plt.subplots(figsize=(8, 5))

# Plot da linha principal (aumentada de 2 para 3)
ax.plot(all_priors, color='black', linewidth=3, label="Estimated Prevalence")

# Linha horizontal da prevalência real (aumentada de 1.5 para 2.5)
ax.axhline(y=test_prev, color='red', linestyle='--', linewidth=2.5, label="True Prevalence")

# Configurações dos eixos (aumentadas de 11 para 14)
ax.set_xlabel("Iterations", fontsize=14)
ax.set_ylabel("Estimated Prevalence", fontsize=14)
ax.set_title("EMQ Prevalence Estimation Over Iterations", fontsize=16, pad=15)

# Aumentar tamanho dos números nos eixos
ax.tick_params(axis='both', which='major', labelsize=12)

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Legenda (aumentar tamanho da fonte)
ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', fontsize=12)

# Bordas
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig("docs/source/images/expectation-maximization.png", dpi=300)