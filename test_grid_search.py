from mlquantify.model_selection import GridSearchQ
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

grid = {
    "__learner_min_samples_split": [3, 8],
    "learner": [RandomForestClassifier(), DecisionTreeClassifier()],
    "measure": ["hellinger", "topsoe"],
}

grid_search = GridSearchQ(
    quantifier=DyS,
    param_grid=grid,
    samples_sizes=42,
    verbose=True
)



grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params)
print("Best Score:", grid_search.best_score)