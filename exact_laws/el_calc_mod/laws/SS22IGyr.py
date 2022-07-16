from typing import List

from .abstract_law import AbstractLaw

class Ss22iGyr(AbstractLaw):
    def __init__(self):
        self.terms = ["source_dpan"]
        pass
    
    def terms_and_coeffs(self,physical_params):
        coeffs = {} 
        coeffs["source_dpan"] = - 1 / 4
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)

def load():
    return Ss22iGyr()
