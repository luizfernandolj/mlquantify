.. _likelihood:

.. currentmodule:: mlquantify.likelihood

===============================
Likelihood-Based Quantification
===============================

Likelihood-based methods (Maximum Likelihood) aim to estimate class prevalences in the test set :math:`U`, assuming that the class distribution (priors) has changed, but the probability densities within each class (:math:`P(X|Y)`) have remained the same (Prior Probability Shift).

Maximum Likelihood Prevalence Estimation (MLPE)
===============================================

The **Maximum Likelihood Prevalence Estimation (MLPE)**, defined in :class:`MLPE`, is the simplest strategy and is considered a trivial starting point or baseline. It naively assumes that the class distribution in the test set (:math:`U`) is the same as in the training set (:math:`L`).

MLPE is not a "true" quantification method but rather a trivial strategy. It simply takes the observed prevalence in the training set and uses it as the estimate for the test set.
If there were no dataset shift (change in distribution), MLPE would be the optimal quantification strategy.

MLPE defines the estimated prevalence in the test set :math:`U` (:math:`\hat{p}^U_{MLPE}(y)`) as the prevalence :math:`p_L(y)` in the training set :math:`L`:

.. math::

    \hat{p}^U_{MLPE}(y) = p_L(y)

**Example**

.. code-block:: python

   from mlquantify.likelihood import MLPE
   
   # MLPE simply returns the training prevalence
   q = MLPE()
   q.fit(X_train, y_train)
   q.predict(X_test) 
   # -> returns prevalence(y_train)


Expectation Maximization for Quantification (EMQ)
=================================================

The **Expectation Maximization for Quantification (EMQ)**, defined in :class:`EMQ` (also known as **SLD** — Saerens, Latinne, Decaestecker) [1]_, is an transductive algorithm that uses a transductive correction of posterior probabilities to estimate class prevalences in the test set :math:`U` by maximizing the likelihood of the observed data [2]_.

The SLD algorithm is based on the Expectation-Maximization (EM) framework, which is an iterative method for finding maximum likelihood estimates in models with latent variables. The :class:`EMQ` works by:

- **Adjusting classifier outputs**: It adjusts the outputs of a probabilistic classifier to correspond to new prior probabilities (prevalences) without the need to retrain the classification model. As a byproduct of this process, it also estimates the new prior probabilities.
- **Iterative refinement**: EMQ is a mutually recursive process that iterates by incrementally updating posterior probabilities (**E-Step**) and then class prevalences (**M-Step**) until the process converges.
- **Convergence guarantee**: The algorithm converges to a global maximum of the likelihood estimate, as the likelihood function is concave and bounded.

The method starts at **Iteration 0**, where the initial estimated prevalence :math:`\hat{p}^{(0)}_U(y)` is defined as the training set prevalence :math:`p_L(y)` (i.e., the MLPE estimate, or priors). From there, EMQ uses iteration to adjust this initial estimate.

.. dropdown:: Mathematical details - EMQ Algorithm

   EMQ iterates between the E and M steps, based on:

   - :math:`\hat{p}^{(s)}_U(\omega_i)`: Estimated prevalence of class :math:`\omega_i` at iteration :math:`s`.
   - :math:`\hat{p}_L(\omega_i)`: Prior probability of class :math:`\omega_i` in the source domain (training).
   - :math:`\hat{p}_L(\omega_i \mid x_k)`: Posterior probability of :math:`x_k` belonging to class :math:`\omega_i`, provided by the calibrated classifier.

   **Initialization (Iteration s=0)**

   For each class :math:`y \in Y`:

   .. math::

      \hat{p}^{(0)}_U(y) \leftarrow p_L(y)

   **E-Step (Expectation) - Posterior Probability Correction**

   Calculates the corrected posterior probability, :math:`p^{(s)}(\omega_i \mid x_k)`. This step adjusts the classifier output probabilities using the ratio between the new estimated prevalence and the training prevalence:

   .. math::

      p^{(s)}(\omega_i \mid x_k) \leftarrow \frac{ \frac{\hat{p}^{(s-1)}_U(\omega_i)}{\hat{p}_L(\omega_i)} \cdot p^{(0)}(\omega_i \mid x_k) }{ \sum_{\omega_j \in Y} \frac{\hat{p}^{(s-1)}_U(\omega_j)}{\hat{p}_L(\omega_j)} \cdot p^{(0)}(\omega_j \mid x_k) }

   **M-Step (Maximization) - Prevalence Update**

   The new prevalence estimate (:math:`\hat{p}^{(s)}_U(\omega_i)`) is the average of the corrected posterior probabilities over all :math:`N` samples in the test set :math:`U`:

   .. math::

      \hat{p}^{(s)}_U(\omega_i) \leftarrow \frac{1}{|U|} \sum_{x_k \in U} p^{(s)}(\omega_i \mid x_k)

   The EMQ iterates the E and M steps until the prevalence parameters converge [1]_ [2]_.


**Example**

.. code-block:: python

   from mlquantify.likelihood import EMQ
   from sklearn.linear_model import LogisticRegression

   # EMQ requires a probabilistic classifier (soft classifier)
   q = EMQ(learner=LogisticRegression())
   q.fit(X_train, y_train)
   
   # Updates predictions based on the test distribution iteratively
   q.predict(X_test) 
   # -> adjusted prevalence dictionary

.. dropdown:: References

   .. [1] Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. Neural computation, 14(1), 21-41.
   .. [2] Esuli, A., Fabris, A., Moreo, A., & Sebastiani, F. (n.d.). Learning to Quantify The Information Retrieval Series.