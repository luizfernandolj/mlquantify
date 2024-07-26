from quantifyML.methods import *
from quantifyML.classfication import *
from quantifyML.utils import get_real_prev
from quantifyML.evaluation import *

import pandas as person
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from quapy.data.base import LabelledCollection
#from quapy.method.aggregative import T50
import time

df = person.read_csv("data/click-prediction.csv")

#df["class"] = df["class"].replace(2, 0)

X = df.drop("class", axis=1)
Y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=64)

clf = RandomForestClassifier(n_estimators=200, random_state=69, n_jobs=-1)
#clf = PWKCLF(alpha=10,n_neighbors=100, n_jobs=-1)

data = LabelledCollection(X_train, y_train)


real_prevalences = get_real_prev(y_test)

quantifier = DySsyn(clf)


start = time.time()
quantifier.fit(X_train.values, y_train.values)
result = quantifier.predict(X_test.values)
#distance = quantifier.distance
#print(distance)
end = time.time()

total_time = end-start

results = person.DataFrame([real_prevalences, result])
results.index = ["real", "pred"]
print(results)
print(f"time: {total_time} seconds")

print(bias(real_prevalences, result))