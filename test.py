from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


from mlquantify.adjust_counting import (
    PCC, 
    CC, 
    FM, 
    GAC,
    GPAC,
    ACC,
    X_method,
    MAX,
    T50,
    MS,
    MS2,
)
from mlquantify.likelihood import (
    EMQ,
    MLPE,
    CDE
)
from mlquantify.mixture import (
    DyS,
    HDy,
    SMM,
    SORD,
    HDx
)
from mlquantify.neighbors import (
    KDEyCS,
    KDEyHD,
    KDEyML,
    PWK
)
from mlquantify.meta import (
    EnsembleQ,
    AggregativeBootstrap,
    QuaDapt
)


from mlquantify.model_selection import (
    BaseProtocol,
    APP,
    NPP,
    UPP,
    PPP
)

from mlquantify.utils import get_prev_from_labels
from mlquantify.metrics import (
    NMD,
    RNOD,
    VSE,
    CvM_L1,
    AE,
    SE,
    MAE,
    MSE,
    KLD,
    RAE,
    NAE,
    NRAE,
    NKLD,
)

# Carregar o dataset Iris
data = load_breast_cancer()
X, y = data.data, data.target

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# Handle both numpy arrays and pandas Series/DataFrames
if isinstance(y_train, pd.Series):
    y_train = y_train.replace({0: 'malignant', 1: 'benign'})
else:
    y_train = np.where(y_train == 0, 'malignant', np.where(y_train == 1, 'benign', y_train))

if isinstance(y_test, pd.Series):
    y_test = y_test.replace({0: 'malignant', 1: 'benign'})
else:
    y_test = np.where(y_test == 0, 'malignant', np.where(y_test == 1, 'benign', y_test))


# Criar e treinar o modelo
rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
rf_train_pred = rf.predict_proba(X_train)
rf_pred = rf.predict_proba(X_test)

quantifier = AggregativeBootstrap

#Usar o quantificador sem learner
# quantifier1 = quantifier()
# predictions1 = quantifier1.aggregate(rf_pred, rf_train_pred, y_train)
# print("Predicted class prevalences 1:", predictions1)

# Usar o quantificador com learner
quantifier2 = quantifier(quantifier=PCC(rf))
quantifier2.fit(X_train, y_train)
predictions2 = quantifier2.predict(X_test)
print("Predicted class prevalences 2:", predictions2)


# Obter a prevalência real das classes no conjunto de teste
real_prevs = get_prev_from_labels(y_test)
real_train_prevs = get_prev_from_labels(y_train)

print("\n--- Metrics for Quantifier ---")
print("MAE 2: ", MAE(real_prevs, predictions2))
print("MSE 2: ", MSE(real_prevs, predictions2))
print("KLD 2: ", KLD(real_prevs, predictions2))
print("AE 2: ", AE(real_prevs, predictions2))
print("SE 2: ", SE(real_prevs, predictions2))
print("RAE 2: ", RAE(real_prevs, predictions2))
print("NAE 2: ", NAE(real_prevs, predictions2))
print("NRAE 2: ", NRAE(real_prevs, predictions2))
print("NKLD 2: ", NKLD(real_prevs, predictions2))
print("NMD 2: ", NMD(real_prevs, predictions2))
print("RNOD 2: ", RNOD(real_prevs, predictions2))
print("VSE 2: ", VSE(real_prevs, predictions2, real_train_prevs))
print("CvM_L1 2: ", CvM_L1(real_prevs, predictions2))


app = APP(batch_size=10,
          n_prevalences=10,
          repeats=1,
          random_state=42)
npp = NPP(batch_size=10,
          n_samples=10,
          random_state=42)
upp = UPP(batch_size=10,
          n_prevalences=20,
          repeats=1,
          random_state=42)
ppp = PPP(batch_size=10,
          prevalences=[0.5, 0.6],
          repeats=1,
          random_state=42)

print("\n--- Batches Prevalence ---\n")

print("====== APP ======")
for idx in app.split(X_test, y_test):
    X_batch, y_batch = X_test.iloc[idx], y_test.iloc[idx]
    print(get_prev_from_labels(y_batch))
    
print("\n====== NPP ======")

for idx in npp.split(X_test, y_test):
    X_batch, y_batch = X_test.iloc[idx], y_test.iloc[idx]
    print(get_prev_from_labels(y_batch))
    
print("\n====== UPP ======")
    
for idx in upp.split(X_test, y_test):
    X_batch, y_batch = X_test.iloc[idx], y_test.iloc[idx]
    print(get_prev_from_labels(y_batch))
    
print("\n====== PPP ======")
for idx in ppp.split(X_test, y_test):
    X_batch, y_batch = X_test.iloc[idx], y_test.iloc[idx]
    print(get_prev_from_labels(y_batch))
