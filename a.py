from libquant.methods.aggregative import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/insects.csv")
#df.replace(2, 0, inplace=True)
#X = df.drop("class", axis=1)
#Y = df["class"]

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=64)

rfc = RandomForestClassifier()

print("Real proportion: ", y_test.value_counts(normalize=True))

quantifier = PCC(classifier=rfc, random_state=32, n_jobs=-1)


quantifier.fit(X_train, y_train)
result = quantifier.estimate(X_test)

print(result)