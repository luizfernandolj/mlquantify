.. _model_selection_evaluation:

.. _model_selection:

Model Selection and Evaluation
------------------------------

Evaluating and selecting quantification models requires dedicated protocols
and metrics — standard classification tools (train/test split, accuracy, F1)
are not sufficient because quantification performance depends on the
**prevalence distribution** of the test data, not just individual labels.

This section covers the full evaluation workflow:

1. **Protocols** — how to generate many test samples with varying prevalences
   from a single dataset (:class:`~mlquantify.model_selection.APP`,
   :class:`~mlquantify.model_selection.UPP`,
   :class:`~mlquantify.model_selection.NPP`).
2. **Hyperparameter tuning** — how to use ``GridSearchQ`` to select the best
   quantifier configuration.
3. **Evaluation metrics** — which error measure to use and when.

**Quick example — full evaluation pipeline:**

.. code-block:: python

   from mlquantify.likelihood import EMQ
   from mlquantify.model_selection import APP
   from mlquantify.metrics import MAE
   from mlquantify.utils import get_prev_from_labels
   from sklearn.linear_model import LogisticRegression
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   import numpy as np

   X, y = make_classification(n_samples=2000, weights=[0.7, 0.3],
                              random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.5, random_state=42)

   q = EMQ(LogisticRegression())
   q.fit(X_train, y_train)

   protocol = APP(batch_size=100, n_prevalences=21, repeats=10,
                  random_state=42)

   errors = []
   for idx in protocol.split(X_test, y_test):
       X_s, y_s = X_test[idx], y_test[idx]
       tp = get_prev_from_labels(y_s)
       pp = q.predict(X_s)
       errors.append(MAE(tp, pp))

   print(f"Mean MAE: {np.mean(errors):.4f} ± {np.std(errors):.4f}")

.. toctree::
   :maxdepth: 2

   Evaluation Protocols <modules/protocols.rst>
   Hyperparameter Tuning <modules/tuning_hyperparameters.rst>
   Evaluation Metrics <modules/evaluation_metrics.rst>