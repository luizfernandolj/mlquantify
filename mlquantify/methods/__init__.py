from .aggregative import * 
from .meta import *
from .non_aggregative import *


AGGREGATIVE = {
    "CC": CC,
    "PCC": PCC,
    "EMQ": EMQ,
    "FM": FM,
    "GAC": GAC,
    "GPAC": GPAC,
    "PWK": PWK,
    "ACC": ACC,
    "MAX": MAX,
    "MS": MS,
    "MS2": MS2,
    "PACC": PACC,
    "T50": T50,
    "X": X_method,
    "DyS": DyS,
    "DySsyn": DySsyn,
    "HDy": HDy,
    "SMM": SMM,
    "SORD": SORD,
}

NON_AGGREGATIVE = {
    "HDx": HDx
}

META = {
    "ENSEMBLE": Ensemble
}


METHODS = AGGREGATIVE | NON_AGGREGATIVE | META
