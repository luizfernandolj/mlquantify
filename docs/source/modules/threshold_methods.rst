.. _threshold_methods:

Threshold Methods
-----------------

The threshold methods are also binary quantifiers (i.e., multiclass problems have not been implemented yet). Proposed by Forman (`2005`_, `2008`_), these algorithms work by adjusting the outputs of a classifier to obtain the class distribution of a test set. Most methods uses a table of thresholds (i.e. 0.0, 0.1, 0.2, ..., 1.0) along with the TPR (True Positive Rate) and FPR (False Positive Rate) to estimate the class distribution of the test set. Each different quantifier uses these values in a different way.

.. _2005:
   https://link.springer.com/chapter/10.1007/11564096_55
.. _2008:
   https://link.springer.com/article/10.1007/s10618-008-0097-y


   
The library implements the following threshold methods:

.. list-table:: Implemented Threshold Methods
    :header-rows: 1

    * - quantifier
      - class
      - reference
    * - Adjusted Classify and Count or Adjusted Count
      - `ACC <generated/mlquantify.methods.aggregative.ACC.html>`_
      - `Forman (2005) <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_
    * - Threshold MAX
      - `MAX <generated/mlquantify.methods.aggregative.MAX.html>`_
      - `Forman (2008) <https://link.springer.com/chapter/10.1007/11564096_55>`_
    * - Median Sweep
      - `MS <generated/mlquantify.methods.aggregative.MS.html>`_
      - `Forman (2005) <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_
    * - Median Sweep 2
      - `MS2 <generated/mlquantify.methods.aggregative.MS2.html>`_
      - `Forman (2005) <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_
    * - Probabilistic Adjusted Classify and Count
      - `PACC <generated/mlquantify.methods.aggregative.PACC.html>`_
      - `Bella et al. (2010) <https://ieeexplore.ieee.org/abstract/document/5694031>`_
    * - Threshold 50
      - `T50 <generated/mlquantify.methods.aggregative.T50.html>`_
      - `Forman (2005) <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_
    * - Threshold X
      - `X_method <generated/mlquantify.methods.aggregative.X_method.html>`_
      - `Forman (2005) <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_
  
You can compute the table of tpr and fpr for each threshold value using the :func:`~mlquantify.utils.method.adjust_threshold` function. This function takes as input the true labels and the predicted probabilities of the positive class generated via cross validation, and returns a table with the thresholds, TPR, and FPR values.

.. code-block:: python

    from mlquantify.utils.method import adjust_threshold, get_scores
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    import numpy as np

    X = np.random.rand(200, 10)
    y = np.random.randint(0, 2, size=200) # Random binary labels [0, 1]
    classes = np.unique(y)

    model = LogisticRegression() # Example model, replace with your own

    # Fit the model to the data via cross validation
    y_labels, probabilities = get_scores(X=X, y=y, learner=model, folds=10, learner_fitted=False)
    probabilities = probabilities[:, 1] # Get the probabilities for the positive class

    thresholds, tprs, fprs = adjust_threshold(y=y_labels, probabilities=probabilities, classes=classes)

    table = pd.DataFrame({
    "Threshold": thresholds,
    "TPR": tprs,
    "FPR": fprs
    })

    print(table)

Or use the :func:`~mlquantify.utils.method.compute_table` with the :func:`~mlquantify.utils.method.compute_tpr` and :func:`~mlquantify.utils.method.compute_fpr` function to get the TPR and FPR values manually:

.. code-block:: python

    from mlquantify.utils.method import compute_table, compute_tpr, compute_fpr
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    import numpy as np

    X = np.random.rand(200, 10)
    y = np.random.randint(0, 2, size=200) # Random binary labels [0, 1]
    classes = np.unique(y)

    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    TP, FP, FN, TN = compute_table(y, y_pred, classes)
    tpr = compute_tpr(TP, FN)
    fpr = compute_fpr(FP, TN)
    print("True Positive Rate (TPR):", tpr)
    print("False Positive Rate (FPR):", fpr)

  