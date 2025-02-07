"mlquantify, a Python package for quantification"

import pandas

from . import base
from . import model_selection
from . import plots
from . import classification
from . import evaluation
from . import methods
from . import utils

ARGUMENTS_SETTED = False

arguments = {
    "y_pred": None,
    "posteriors_train": None,
    "posteriors_test": None,
    "y_labels": None,
    "y_pred_train": None,
}

def set_arguments(y_pred=None, posteriors_train=None, posteriors_test=None,  y_labels=None, y_pred_train=None):
    global ARGUMENTS_SETTED
    global arguments
    arguments["y_pred"] = y_pred
    arguments["posteriors_train"] = posteriors_train.to_numpy() if isinstance(posteriors_train, pandas.DataFrame) else posteriors_train
    arguments["posteriors_test"] = posteriors_test.to_numpy() if isinstance(posteriors_test, pandas.DataFrame) else posteriors_test
    arguments["y_labels"] = y_labels
    arguments["y_pred_train"] = y_pred_train
    
    ARGUMENTS_SETTED = True