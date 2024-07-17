#from quantifyml.methods.aggregative import *
from quantiML.methods import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from quapy.data.base import LabelledCollection
#from quapy.method.aggregative import T50
import time

df = pd.read_csv("data/UWave.csv")
#df["class"] = df["class"].replace(2, 0)
#X = df.drop("class", axis=1)
#Y = df["class"]


X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=64)

rfc = RandomForestClassifier(n_estimators=200, random_state=69, n_jobs=-1)


data = LabelledCollection(X_train, y_train)

#print("Quapy result:")
#acc = acc(rfc)
#acc.fit(data)
#result = acc.quantify(X_test)
#print(result)


print("Real proportion:")
rp = np.round(y_test.value_counts(normalize=True), 3).to_dict()
print(dict(sorted(rp.items())))

quantifier = MS(rfc)

start = time.time()
quantifier.fit(X_train, y_train, learner_fitted=False, cv_folds=3)
result = quantifier.predict(X_test)
end = time.time()

total_time = end-start
print(F"{result} in {total_time} seconds")