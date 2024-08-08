from quantifyML.methods import *
from quantifyML.classification import *
from quantifyML.utils import *
from quantifyML.evaluation import *
from quantifyML.model_selection import GridSearchQ
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from quapy.data.base import LabelledCollection
import time

# Load and prepare the dataset
df = pd.read_csv("data/click-prediction.csv")
X = df.drop("class", axis=1)
y = df["class"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=64)

# Initialize the classifier
clf = RandomForestClassifier(random_state=69, n_jobs=-1)

# Define the quantifier without grid search
quantifier = PCC(clf)

# Start timing for quantifier without grid search
start_no_gs = time.time()

# Fit the quantifier on the training data
quantifier.fit(X_train.values, y_train.values)

# Make predictions with the quantifier
result_qtf_no_gs = quantifier.predict(X_test.values)

# End timing
end_no_gs = time.time()
total_time_no_gs = end_no_gs - start_no_gs

# Print real prevalences from the test set
real_prevalences = get_real_prev(y_test)

# Initialize the parameter grid for GridSearchQ
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Initialize GridSearchQ
grid_search = GridSearchQ(
    model=quantifier,
    param_grid=param_grid,
    protocol='app',  # Artificial Prevalence Protocol
    n_prevs=10,
    scoring="ae",  # Mean Absolute Error
    refit=True,
    n_jobs=-1,
    verbose=True
)

# Start timing for grid search
start_gs = time.time()

# Fit the GridSearchQ on the training data
grid_search.fit(X_train.values, y_train.values)

# Make predictions with the best model found by GridSearchQ
best_model = grid_search.best_model()
result_qtf_gs = best_model.predict(X_test.values)

# End timing
end_gs = time.time()
total_time_gs = end_gs - start_gs

# Create a DataFrame to store the results
results = pd.DataFrame({
    "real": real_prevalences,
    "quantifier_no_gs": result_qtf_no_gs,
    "quantifier_gs": result_qtf_gs
})

# Print the results
print(results)
print(f"Time taken without grid search: {total_time_no_gs:.2f} seconds")
print(f"Time taken with grid search: {total_time_gs:.2f} seconds")

# Print the errors
error_no_gs = absolute_error(real_prevalences, result_qtf_no_gs)
error_gs = absolute_error(real_prevalences, result_qtf_gs)
print(f"Quantifier error without grid search: {error_no_gs}")
print(f"Quantifier error with grid search: {error_gs}")
