.. _quantification_protocols:

.. currentmodule:: mlquantify.protocols

==============================
Protocols for Quantification
==============================

Quantification protocols are designed to evaluate quantifiers by generating multiple test samples with varying class prevalences. These protocols ensure robust assessment of quantification methods under different distributional shifts.

Experimental evaluation primarily uses two main protocols:

Artificial-Prevalence Protocol (APP)
====================================

The :class:`APP` is the most commonly used protocol, leveraging widely available classification datasets to artificially vary class prevalences in test samples.

- Generates multiple test samples by subsampling the original test set to produce varying class prevalences.
- Simulates prior probability shift (:math:`P_L(Y) \neq P_U(Y)`) while maintaining conditional feature distributions constant.
- Allows creation of extensive test points from a single dataset for thorough evaluation.

**Example**

.. code-block:: python

    from mlquantify.protocols import APP
    from mlquantify.utils import get_prev_from_labels

    # Initialize protocol
    app = APP(
        batch_size=[100, 200], 
        n_prevalences=5, 
        repeats=3, 
        random_state=42
    )

    for idx in app.split(X_test, y_test):
        X_sample, y_sample = X_test[idx], y_test[idx]
        real_prevalence = get_prev_from_labels(y_sample)
        # Evaluate quantifier on (X_sample, y_sample)


Natural-Prevalence Protocol (NPP)
=================================

The NPP uses naturally occurring prevalence variations by partitioning a large test set into random sub-samples, preserving their inherent class distributions.

- Preserves real-world prevalence distributions without artificial manipulation.
- Provides realistic evaluation of quantifiers but is less common due to data requirements.

**Example**

.. code-block:: python

    from mlquantify.protocols import NPP
    from mlquantify.utils import get_prev_from_labels

    # Initialize protocol
    npp = NPP(batch_size=100, random_state=42)

    for idx in npp.split(X_test, y_test):
        X_sample, y_sample = X_test[idx], y_test[idx]
        real_prevalence = get_prev_from_labels(y_sample)
        # Evaluate quantifier on (X_sample, y_sample)



Uniform Prevalence Protocol (UPP)
=================================

The :class:`UPP` is a variant of the APP that ensures uniform sampling of class prevalences across the entire range [0, 1].

- Guarantees that all possible prevalence values are equally represented in the test samples.
- Useful for comprehensive evaluation of quantifiers across the full prevalence spectrum.
- Particularly beneficial in multiclass quantification tasks (less computationally intensive).

**Example**

.. code-block:: python

    from mlquantify.protocols import UPP
    from mlquantify.utils import get_prev_from_labels

    # Initialize protocol
    upp = UPP(
        batch_size=[100, 200], 
        n_prevalences=5, 
        repeats=3, 
        random_state=42
    )

    for idx in upp.split(X_test, y_test):
        X_sample, y_sample = X_test[idx], y_test[idx]
        real_prevalence = get_prev_from_labels(y_sample)
        # Evaluate quantifier on (X_sample, y_sample)


Personalized Prevalence Protocol (PPP)
======================================

The :class:`PPP` is another APP variant that allows users to specify desired class prevalences for generating test samples, since APP sample all possible prevalences uniformly.

- Enables targeted evaluation of quantifiers at specific prevalence levels.
- Useful for scenarios where certain prevalence values are of particular interest.

**Example**

.. code-block:: python

    from mlquantify.protocols import PPP
    from mlquantify.utils import get_prev_from_labels

    # Initialize protocol with desired prevalences
    ppp = PPP(batch_size=100, prevalences=[0.1, 0.9], repeats=3, random_state=42)

    for idx in ppp.split(X_test, y_test):
        X_sample, y_sample = X_test[idx], y_test[idx]
        real_prevalence = get_prev_from_labels(y_sample)
        # Evaluate quantifier on (X_sample, y_sample)


References
==========

.. [1] Esuli, A., Fabris, A., Moreo, A., & Sebastiani, F. (n.d.). Learning to Quantify The Information Retrieval Series.