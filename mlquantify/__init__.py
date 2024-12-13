"mlquantify, a Python package for quantification"

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
    "posteriors": None,
    "y_labels": None,
    "y_train_pred": None,
}

def set_arguments(y_pred=None, posteriors=None, y_labels=None, y_train_pred=None):
    global ARGUMENTS_SETTED
    global arguments
    arguments["y_pred"] = y_pred
    arguments["posteriors"] = posteriors
    arguments["y_labels"] = y_labels
    arguments["y_train_pred"] = y_train_pred
    ARGUMENTS_SETTED = True
