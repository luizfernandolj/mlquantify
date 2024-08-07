from quantifyML.methods import *
from quantifyML.classfication import *
from quantifyML.utils import get_real_prev
from quantifyML.evaluation import *

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

clf = RandomForestClassifier(n_estimators=200, random_state=69, n_jobs=-1)
#clf = PWKCLF(alpha=10,n_neighbors=100, n_jobs=-1)


real_prevalences = get_real_prev(y_test)

app = APP(learner=clf,
          models="CC", 
          batch_size=[5000, 1000],
          n_prevs=10,
          n_jobs=-1,
          return_type="table",
          measures=["ae", "rae"],
          verbose=True)

app.fit(X_train.values, y_train.values)

table = app.predict(X_test.values, y_test.values)

print(table.round(3))
