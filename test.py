
import mlquantify as mq
from mlquantify.evaluation.protocol import APP
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Loading dataset from sklearn
features, target = load_breast_cancer(return_X_y=True)

#Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)

app = APP(models=["CC", "EMQ", "DyS"],
          batch_size=[10, 50, 100],
          learner=RandomForestClassifier(),
          n_prevs=100, # Default
          n_jobs=-1,
          return_type="table",
          measures=["ae", "se"],
          verbose=True)

app.fit(X_train, y_train)

table = app.predict(X_test, y_test)

print(table)