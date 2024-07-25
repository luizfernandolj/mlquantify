from .aggregative import *

METHODS = {
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
    "HDy": HDy,
    "SMM": SMM,
    "SORD": SORD,
}

def get_class(method):
    return METHODS.get(method)