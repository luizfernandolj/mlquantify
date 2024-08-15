
import mlquantify as mq
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

features, target = fetch_covtype(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5)

model = RandomForestClassifier(n_jobs=-1)
model.fit(X_train, y_train)

scores = model.predict_proba(X_test)

mq.plots.class_distribution_plot(values=scores,
                                 labels=y_test,
                                 bins=30, # Default
                                 title="Scores Distribution",
                                 legend=True,
                                 save_path="dist.png")

