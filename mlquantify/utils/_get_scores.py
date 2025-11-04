import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

def apply_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv= 5,
    function= 'predict_proba',
    stratified= True,
    random_state= None,
    shuffle= True):
    """
    Perform cross-validation and return predictions with true labels for each fold.
    
    Parameters:
    -----------
    model : estimator
        Model with fit and predict/predict_proba methods
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    cv : int, default=5
        Number of cross-validation folds
    function : str, default='predict_proba'
        Method to use for predictions ('predict' or 'predict_proba' or any callable)
    stratified : bool, default=True
        Whether to use stratified cross-validation
    random_state : int or None, default=None
        Random state for reproducibility
    shuffle : bool, default=True
        Whether to shuffle data before splitting
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        predictions, true_labels for all folds
    """
    
    # Choose cross-validation strategy
    if stratified:
        cv_splitter = StratifiedKFold(
            n_splits=cv, 
            shuffle=shuffle, 
            random_state=random_state
        )
    else:
        cv_splitter = KFold(
            n_splits=cv, 
            shuffle=shuffle, 
            random_state=random_state
        )
    
    # Pre-allocate arrays
    all_predictions = []
    all_true_labels = []
    
    # Perform cross-validation
    for train_idx, test_idx in cv_splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        if type(function) is str:
            if not hasattr(model, function):
                raise AttributeError(f"The model does not have the method '{function}'.")
            predictions = getattr(model, function)(X_test)
        elif callable(function):
            predictions = function(X_test)
        else:
            raise ValueError("The 'function' parameter must be a string or a callable.")
        
        all_predictions.append(predictions)
        all_true_labels.append(y_test)
    
    # Concatenate all predictions and labels
    final_predictions = np.vstack(all_predictions) if function == 'predict_proba' else np.concatenate(all_predictions)
    final_true_labels = np.concatenate(all_true_labels)
    
    return final_predictions, final_true_labels



def apply_bootstrap(
    model,
    X: np.ndarray,
    y: np.ndarray = None,
    n_bootstraps: int = 100,
    function: str = 'predict_proba',
    random_state: int = None
):
    """
    Perform bootstrap resampling and return predictions with true labels for each bootstrap sample.
    If y is None, bootstrap and fit using only X, and check if model is fitted.
    
    Parameters:
    -----------
    model : estimator
        Model with fit and predict/predict_proba methods
    X : np.ndarray
        Feature matrix
    y : np.ndarray or None
        Target vector (optional)
    n_bootstraps : int, default=100
        Number of bootstrap samples
    function : str or callable, default='predict_proba'
        Method to use for predictions ('predict' or 'predict_proba' or any callable)
    random_state : int or None, default=None
        Random state for reproducibility
    """

    if random_state is not None:
        np.random.seed(random_state)

    all_predictions = []
    all_true_labels = [] if y is not None else None

    for _ in range(n_bootstraps):
        bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
        X_bootstrap = X[bootstrap_indices]
        if y is not None:
            y_bootstrap = y[bootstrap_indices]
            model.fit(X_bootstrap, y_bootstrap)
        else:
            model.fit(X_bootstrap)

            # Check if model is fitted - raise error if not
            if not hasattr(model, "fitted_") or not getattr(model, "fitted_"):
                # Some models have other indicators, as a fallback we could just pass
                # or do other checks. For simplicity, we check fitted_
                # If not fitted, raise exception
                raise ValueError("Model does not appear to be fitted after fit(X).")

        if type(function) is str:
            if not hasattr(model, function):
                raise AttributeError(f"The model does not have the method '{function}'.")
            predictions = getattr(model, function)(X_bootstrap)
        elif callable(function):
            predictions = function(X_bootstrap)
        else:
            raise ValueError("The 'function' parameter must be a string or a callable.")

        all_predictions.append(predictions)
        if y is not None:
            all_true_labels.append(y_bootstrap)

    if function == 'predict_proba':
        final_predictions = np.vstack(all_predictions)
    else:
        final_predictions = np.concatenate(all_predictions)

    if y is not None:
        final_true_labels = np.concatenate(all_true_labels)
        return final_predictions, final_true_labels
    else:
        return final_predictions