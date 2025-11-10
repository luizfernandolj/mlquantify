from joblib import load


def load_quantifier(path:str):
    """
    Load a quantifier from a file.
    
    Parameters
    ----------
    path : str
        Path to the file containing the quantifier.
    
    Returns
    -------
    Quantifier
        Loaded quantifier.
    """
    return load(path)
