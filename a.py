from quantifyml.methods.aggregative import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/BNG.csv")
df["class"] = df["class"].replace(2, 0)
X = df.drop("class", axis=1)
Y = df["class"]

#X = df.iloc[:, :-1]
#Y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=64)

rfc = RandomForestClassifier(n_estimators=200, random_state=69, n_jobs=-1)

print("Real proportion:")
print(np.round(y_test.value_counts(normalize=True), 3).to_dict())

quantifier = PACC(classifier=rfc)


quantifier.fit(X_train, y_train)
result = quantifier.predict(X_test)

print(result)