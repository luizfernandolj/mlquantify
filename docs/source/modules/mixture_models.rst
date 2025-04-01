.. _mixture_models:

Mixture Models
--------------

Mixture models are binary quantification methods that assume the cumulative distribution of an unknown dataset, or test set, is a mixture of two or more distributions. This concept was first introduced by Forman (`2005`_, `2008`_).

.. _2005:
   https://link.springer.com/chapter/10.1007/11564096_55
.. _2008:
   https://link.springer.com/article/10.1007/s10618-008-0097-y

The base structure of mixture models uses the scores of the training set generated via cross-validation, and combines them to approximate the distribution of the test set. The quantification process is performed by estimating the parameters of the mixture using the training set scores and then applying the model to the test set scores.

The library implements the following mixture models:

.. list-table:: Implemented Mixture Models
    :header-rows: 1

    * - quantifier
      - class
      - reference
    * - Distribution y-Similarity
      - `DyS <generated/mlquantify.methods.aggregative.DyS.html>`_
      - `Maletzke et al. (2019) <https://ojs.aaai.org/index.php/AAAI/article/view/4376>`_
    * - Synthetic Distribution y-Similarity
      - `DySsyn <generated/mlquantify.methods.aggregative.DySsyn.html>`_
      - `Maletzke et al. (2021) <https://ieeexplore.ieee.org/abstract/document/9679104>`_
    * - Hellinger Distance Minimization
      - `HDy <generated/mlquantify.methods.aggregative.HDy.html>`_
      - `González-Castro et al. (2013) <https://www.sciencedirect.com/science/article/abs/pii/S0020025512004069?casa_token=W6UksOigmp4AAAAA:ap8FK5mtpAzG-s8k2ygfRVgdIBYDGWjEi70ueJ546coP9F-VNaCKE5W_gsAv0bWQiwzt2QoAuLjP>`_
    * - Sample Mean Matching
      - `SMM <generated/mlquantify.methods.aggregative.SMM.html>`_
      - `Hassan et al. (2013) <https://ieeexplore.ieee.org/document/9260028>`_
    * - Sample Ordinal Distance
      - `SORD <generated/mlquantify.methods.aggregative.SORD.html>`_
      - `Maletzke et al. (2019) <https://ojs.aaai.org/index.php/AAAI/article/view/4376>`_
  

Some algorithms such as DyS, Dyssyn and HDy are based on distances between the mixture of the train scores and the test scores. Hdy for example, uses the Hellinger distance to measure the difference between the two distributions. All the distances can be accessed through the `mlquantify.utils.method` module, with 4 different distance functions implemented in the library:

- :func:`~mlquantify.utils.method.sqEuclidean`;
- :func:`~mlquantify.utils.method.probsymm`;
- :func:`~mlquantify.utils.method.topsoe`;
- :func:`~mlquantify.utils.method.hellinger`.

These methods also have the `best_distance` method, which allows you to obtain the best distance computed by the method. Below is an example of how to use this approach:

.. code-block:: python

     from mlquantify.methods import DyS
     from sklearn.linear_model import LogisticRegression
     import numpy as np

     X_train = np.random.rand(100, 10)
     y_train = np.random.randint(0, 2, size=100)
     X_test = np.random.rand(50, 10)

     quantifier = DyS(LogisticRegression())
     quantifier.fit(X_train, y_train)
     distance = quantifier.best_distance(X_test)
     print(distance)