from . import *


def get_values(X, y, clf, scores:bool=False, tprfpr:bool=False):
        values = {}
        
        if scores:
            values["scores"] = get_scores(X, y, clf)
        if tprfpr:
            values["tprfpr"] = get_tprfpr(X, y, clf)
            
        return values