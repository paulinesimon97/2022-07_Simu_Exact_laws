import importlib
import os
from typing import Dict, List, Tuple

from ..terms import TERMS

here = os.path.dirname(os.path.realpath(__file__))


class AbstractLaw:
    def __init__(self):
        pass
    
    def terms_and_coeffs(self, *args, **kwargs) -> Tuple[List[str], Dict[str,float]]:
        raise NotImplementedError("You have to reimplement this method")
    
    def list_variables(self,terms):
        variables = []
        for term in terms:
            variables += TERMS[term].variables()
        return list(set(variables))

    def variables(self) -> List[str]:
        raise NotImplementedError("You have to reimplement this method") 

def load_law(name):
    mod = importlib.import_module(f"exact_laws.exact_laws_calc.laws.{name}", "*")
    return mod.load()


def load_all():
    laws = [f[:-3] for f in os.listdir(here) if f[-3:] == '.py' and f != '__init__.py']
    return {law: load_law(law) for law in laws}


LAWS = load_all()
