.. _aggregative_quantification:

Aggregative Quantification
--------------------------

Aggregative quantification methods estimate class prevalences by aggregating
predictions made on individual instances. All aggregative methods share a
common three-step structure:

1. **fit** — the model is trained on labelled data. The underlying classifier
   (called the *estimator*) is fitted, and a *representation* of the training
   class distributions is computed (confusion matrix, score histograms, density
   estimates, …).
2. **predict** — the trained model generates predictions for each individual
   item in the unlabelled test set. These predictions can be hard labels or
   soft probabilities.
3. **aggregate** — the predictions are combined into a single prevalence
   estimate. This is the step that distinguishes quantifiers from plain
   classifiers: it corrects for distributional shift, rather than just
   counting labels.

This separation into ``fit``, ``predict``, and ``aggregate`` mirrors the
scikit-learn API and enables modular reuse of any standard classifier as a
backbone.

.. admonition:: The ``aggregate`` method

   Every aggregative quantifier exposes an :meth:`aggregate` method that
   performs only step 3. This lets you quantify without re-predicting when
   classifier outputs are already available:

   .. code-block:: python

      from mlquantify.likelihood import EMQ
      from sklearn.linear_model import LogisticRegression

      q = EMQ(LogisticRegression())
      q.fit(X_train, y_train)

      # Full pipeline
      prevalences = q.predict(X_test)

      # Or: re-use existing posteriors
      proba = q.estimator_.predict_proba(X_test)
      prevalences = q.aggregate(proba, q.train_predictions_, q.train_labels_)

For the theoretical grounding — why counting alone is biased, what dataset
shift means, and how to choose a method — see :ref:`quantification_foundations`.

.. toctree::
   :maxdepth: 2

   modules/using_aggregative.rst
   Counting Methods <modules/counters.rst>
   Adjusted Counting <modules/adjust_counting.rst>
   Likelihood Methods <modules/likelihood.rst>
   Distribution Matching <modules/distribution_matching.rst>
   Nearest Neighbours <modules/neighbors.rst>
   Quantification Trees <modules/tree.rst>
   Explicit Loss Minimization <modules/elm.rst>

