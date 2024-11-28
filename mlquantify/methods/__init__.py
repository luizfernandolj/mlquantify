from . import aggregative
from . import meta
from . import non_aggregative


AGGREGATIVE = {
    "CC": aggregative.CC,
    "PCC": aggregative.PCC,
    "EMQ": aggregative.EMQ,
    "FM": aggregative.FM,
    "GAC": aggregative.GAC,
    "GPAC": aggregative.GPAC,
    "PWK": aggregative.PWK,
    "ACC": aggregative.ACC,
    "MAX": aggregative.MAX,
    "MS": aggregative.MS,
    "MS2": aggregative.MS2,
    "PACC": aggregative.PACC,
    "T50": aggregative.T50,
    "X": aggregative.X_method,
    "DyS": aggregative.DyS,
    "DySsyn": aggregative.DySsyn,
    "HDy": aggregative.HDy,
    "SMM": aggregative.SMM,
    "SORD": aggregative.SORD,
}

NON_AGGREGATIVE = {
    "HDx": non_aggregative.HDx
}

META = {
    "ENSEMBLE": meta.Ensemble
}


METHODS = AGGREGATIVE | NON_AGGREGATIVE | META
