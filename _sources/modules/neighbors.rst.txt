.. _neighbors_kde:

.. currentmodule:: mlquantify.neighbors

=======================================
Neighbors and Kernel Density Estimation
=======================================

Neighbor-based methods and density estimation techniques (such as KDE) focus on exploring the proximity and topology of data to infer class distribution.

KDEy: Kernel Density Estimation y-Similarity
============================================

**KDEy** is a multi-class quantification approach that conceptually falls under the **Distribution Matching (DM)** methods.

:class:`KDEy` acts as an innovative representation mechanism aimed at overcoming the limitations of traditional DM methods that rely on histograms (like HDy). For multi-class problems, KDEy models the distribution of posterior probabilities (y-scores) using **Kernel Density Estimation (KDE)**, which results in Gaussian Mixture Models (GMMs).

**Advantage**

This multivariate representation, operating on the unit simplex (:math:`\Delta_{n-1}`), allows the model to preserve and utilize correlations and interactions between classes—information often lost in class-by-class histogram methods. The quantification problem is then framed as the task of reconstructing the test set distribution as a convex linear combination of the training class distributions.

**Performance**

KDEy, particularly its **Maximum Likelihood (KDEy-ML)** variant, has proven to be one of the strongest quantifiers, often outperforming other histogram-based DM models and remaining competitive with EMQ (SLD) [1]_.

The methods can be found in :class:`KDEyML`, :class:`KDEyHD` and :class:`KDEyCS`, following the methods described in [1]_.

.. dropdown:: Mathematical details - KDEy Framework

   Framed within the Distribution Matching framework, KDEy seeks the prevalence vector :math:`\hat{\alpha}` that minimizes the discrepancy :math:`D` between the weighted mixture distribution (:math:`p_\alpha`) and the test set distribution (:math:`q_{\tilde{U}}`).

   **1. Mixture Density Function**

   In KDEy, the mixture probability density (:math:`p_\alpha`) is defined as the weighted sum of the probability densities of each class in the training set (:math:`p_{\tilde{L}_i}`), where :math:`\alpha = (\alpha_1, \alpha_2, \dots, \alpha_n)` is the prevalence vector being optimized:

   .. math::

      p_\alpha(\tilde{x}) = \sum_{i=1}^n \alpha_i p_{\tilde{L}_i}(\tilde{x})

   Where :math:`p_{\tilde{L}_i}(\tilde{x})` is the Kernel Density Estimator (KDE) for the posterior probabilities of class :math:`i` (represented as :math:`\tilde{L}_i`). The general kernel density estimator :math:`p_\theta(x)` is given by:

   .. math::

      p_\theta(x) = \frac{1}{|X|} \sum_{x_i \in X} K\left(\frac{x - x_i}{h}\right)

   Where :math:`K` is the kernel function (typically Gaussian) and :math:`h` is the bandwidth [1]_.

   **2. Optimization Objective (Distribution Matching)**

   The goal is to find the prevalence vector :math:`\hat{\alpha}` that minimizes the divergence :math:`D` (such as Hellinger Distance or Cauchy-Schwarz) between the mixture distribution :math:`p_\alpha` and the observed distribution in the test set :math:`q_{\tilde{U}}` (which is also a GMM obtained via KDE) [1]_:

   .. math::

      \hat{\alpha} = \operatorname*{arg\,min}_{\alpha \in \Delta_{n-1}} D(p_\alpha || q_{\tilde{U}})

**Example**

.. code-block:: python

   from mlquantify.neighbors import KDEy
   from sklearn.ensemble import RandomForestClassifier

   # KDEy uses kernel density estimation on classifier scores
   q = KDEy(learner=RandomForestClassifier(), bandwidth=0.1)
   q.fit(X_train, y_train)
   q.predict(X_test)

.. dropdown:: References

    .. [1] Moreo, A., González, P., & del Coz, J. J. (2024). Kernel Density Estimation for Multiclass Quantification. http://arxiv.org/abs/2401.00490


PWK: Pair-wise Weighted K-Nearest Neighbors
===========================================

**PWK (Pair-wise Weighted K-Nearest Neighbors)** and its variant **PWK**\ :math:`\alpha` are algorithms based on k-Nearest Neighbors (k-NN), designed for binary quantification [2]_.

:class:`PWK` uses a weighted *k-NN* classification rule to estimate prevalence. Unlike Adjusted Classify & Count (ACC) methods that use error rates from a trained classifier, NN methods explore the **topology** of the data to perform the estimation [2]_.

PWK adjusts predictions to compensate for the systematic bias in favor of the majority class, which is inherent in classifiers trained on imbalanced data.

**Weighting Mechanism**

PWK bases its weighting on class size. Specifically, the weight (:math:`w_c`) of a class :math:`c` is defined to be inversely proportional to the size of that class in the training set.

**PWK**\ :math:`\alpha` introduces an :math:`\alpha` parameter into the weighting, allowing flexibility in bias compensation. PWK and PWK\ :math:`\alpha` are among the most stable and competitive quantifiers tested [2]_.

**Example**

.. code-block:: python

   from mlquantify.neighbors import PWK
   
   # PWK operates directly on the feature space using k-NN
   q = PWK(n_neighbors=10)
   q.fit(X_train, y_train)
   q.predict(X_test)

.. dropdown:: References

    .. [2] Barranquero, J., González, P., Díez, J., & del Coz, J. J. (2013). On the study of nearest neighbor algorithms for prevalence estimation in binary problems. Pattern Recognition, 46(2), 472-482. https://doi.org/10.1016/j.patcog.2012.07.022