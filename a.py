#from quantifyml.methods.aggregative import *
from quantiML.methods import *
from quantiML.utils import get_real_prev
from quantiML.evaluation import *

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from quapy.data.base import LabelledCollection
#from quapy.method.aggregative import T50
import time

df = pd.read_csv("data/click-prediction.csv")

#df["class"] = df["class"].replace(2, 0)

X = df.drop("class", axis=1)
Y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=64)

rfc = RandomForestClassifier(n_estimators=200, random_state=69, n_jobs=-1)


data = LabelledCollection(X_train, y_train)


real_prevalences = get_real_prev(y_test)

quantifier = PCC(rfc)

start = time.time()
quantifier.fit(X_train, y_train, learner_fitted=False, cv_folds=3)
result = quantifier.predict(X_test)
#distance = quantifier.distance
#print(distance)
end = time.time()

total_time = end-start

results = pd.DataFrame([real_prevalences, result])
results.index = ["real", "pred"]
print(results)
print(f"time: {total_time} seconds")

print(normalized_absolute_error(real_prevalences, result))