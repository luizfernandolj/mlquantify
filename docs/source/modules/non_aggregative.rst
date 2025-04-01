.. _non_aggregative:

Non Aggregative Quantification
------------------------------

The case for non-aggregative quantification methods is that they do not rely on the predicted values of a classifier to estimate the class distribution of the test set. Instead, they try to estimate the class distribution directly, using the training set and/or the test set. In this moment the library implements only the :class:`~mlquantify.methods.non_aggregative.HDx` method, which is a non-aggregative binary quantifier based on the idea of estimating the class distribution by using the distance between the training features and the test features. The method is similar to the :class:`~mlquantify.methods.aggregative.HDy` quantifier, proposed by the same author (`González-Castro et al. (2013)`_)

.. _González-Castro et al. (2013):
   https://www.sciencedirect.com/science/article/abs/pii/S0020025512004069?casa_token=W6UksOigmp4AAAAA:ap8FK5mtpAzG-s8k2ygfRVgdIBYDGWjEi70ueJ546coP9F-VNaCKE5W_gsAv0bWQiwzt2QoAuLjP