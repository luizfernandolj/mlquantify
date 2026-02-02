from mlquantify.neighbors import PWK
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

train_df = pd.read_csv("train.csv", index_col=0)
test_df = pd.read_csv("test.csv", index_col=0)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

X_train = train_df.drop("class", axis=1)
y_train = train_df["class"]

X_test = test_df.drop("class", axis=1)
y_test = test_df["class"]

pwk = PWK()
pwk.fit(X_train, y_train)

print(pwk.predict(X_test))