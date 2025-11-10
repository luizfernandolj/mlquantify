import os
from joblib import effective_n_jobs

def resolve_n_jobs(n_jobs=None):
    """Resolve n_jobs like sklearn, with support for -1 and nested contexts."""
    return effective_n_jobs(n_jobs)
