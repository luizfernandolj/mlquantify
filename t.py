import pandas as pd
import mlquantify as ml

from mlquantify.model_selection._protocol import UPP

df = pd.read_csv("Chess game.csv")
df.head(3)

upp = UPP(batch_size=100, n_prevalences=5, random_state=42, repeats=1)

X = df.drop(columns=['game'])
y = df['game']

for idx in upp.split(X, y):
    print(y.iloc[idx])