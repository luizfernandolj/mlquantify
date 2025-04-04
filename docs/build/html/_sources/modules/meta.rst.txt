.. _meta:

===================
Meta Quantification
===================

.. currentmodule:: mlquantify.methods.meta

Meta quantification methods are a class of quantification methods that use the predictions of other quantifiers to estimate the class distribution of the test set. These methods can be seen as a meta-learner that takes the predictions of other quantifiers as input and learns to combine them to produce a more accurate estimate of the class distribution.

mlquantify provides only one meta-quantifier, the :class:`~mlquantify.methods.meta.Ensemble` method, which is a meta-quantifier that uses the predictions of one quantifier several times to estimate the class distribution of the test set. The process for this method is defined as:

- Take the train set and generate samples varying class distribution :math:`S_i`;
- Copy the quantifier several times :math:`M_i`, fitting each one on a different sample;
- Aggregate the predictions of each quantifier on the test set by mean or median.

Other way to use the Ensemble quantifier is dynamically by a criteria, proposed by Pérez-Gállegzo (`2017`_, `2019`_):

- **Training prevalence (ptr)**: runs all models on the test set :math:`U` and ranks them according to the difference between the mean estimated prevalence for :math:`U` and the prevalence in :math:`S_i`
- **Distribution similarity (ds)**: Compares the distribution of posteriors between each sample :math:`S_i` and :math:`U` ranking each quantifier based on the Hellinger distance computed on histograms.

.. _2017:
   https://www.sciencedirect.com/science/article/pii/S1566253516300628?casa_token=XblH-3kwhf4AAAAA:oxNRiCdHZQQa1C8BCJM5PBnFrd26p8-9SSBdm8Luf1Dm35w88w0NdpvoCf1RxBBqtshjyAhNpsDd
.. _2019:
   https://www.sciencedirect.com/science/article/abs/pii/S1566253517303652?casa_token=jWmc592j5uMAAAAA:2YNeZGAGD0NJEMkcO-YBr7Ak-Ik7njLEcG8SKdowLdpbJ0mwPjYKKiqvQ-C3qICG8yU0m4xUZ3Yv

The basic usage of the :class:`~mlquantify.methods.meta.Ensemble` method is as follows:

.. code-block:: python

    from mlquantify.methods import FM, Ensemble
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, size=100)
    X_test = np.random.rand(50, 10)
    y_test = np.random.randint(0, 2, size=50)

    model = FM(RandomForestClassifier())
    ensemble = Ensemble(quantifier=model,
                        size=50,
                        selection_metric='ptr', # Training prevalence
                        return_type='mean',
                        n_jobs=-1,
                        verbose=True)

    ensemble.fit(X_train, y_train)

    predictions = ensemble.predict(X_test)

    print(predictions)