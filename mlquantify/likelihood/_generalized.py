# likelihood/_generalized.py

from mlquantify.base_aggregative import AggregativeMixin, SoftPredictionMixin
from mlquantify.compose import LikelihoodComposeQuantifier
from mlquantify.representations import PredictionRepresentation


class MLPE(SoftPredictionMixin, AggregativeMixin, LikelihoodComposeQuantifier):
    r"""Maximum Likelihood Prior Estimation (MLPE) quantifier.

    Estimates class prevalences by maximising the mixture log-likelihood of
    test posterior probabilities under class-conditional distributions learned
    from training data, using the likelihood-composition framework.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    solver : str, default='slsqp'
        Optimization solver.
    tau_0 : float, default=0.0
        Regularisation weight for the first class.
    tau_1 : float, default=0.0
        Regularisation weight for the second class.
    cv : int or None, default=None
        Cross-validation folds for computing training scores.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.likelihood import MLPE
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = MLPE(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.33, 1: 0.34, 2: 0.33}

    References
    ----------
    .. dropdown:: References

        .. [1] Saerens, M., Latinne, P., & Decaestecker, C. (2002).
               Adjusting the Outputs of a Classifier to New a Priori Probabilities.
               *Neural Computation*, 14(1), 2141–2156.
    """


    def __init__(
        self,
        estimator=None,
        solver="slsqp",
        tau_0=0.0,
        tau_1=0.0,
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            representation=PredictionRepresentation(
                method="soft",
                average=False,
            ),
            solver=solver,
            aggregative=True,
            tau_0=tau_0,
            tau_1=tau_1,
            random_state=random_state,
        )
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
