from typing import List

from .abstract_term import AbstractTerm

class FluxDvdvdv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, values) -> (float or List[float]):
        return self.flux(("Ibx", "Iby", "Ibz"), ("Ibx", "Iby", "Ibz"), ("vx", "vy", "vz"), datadic=values)

    def variables(self) -> List[str]:
        return ['Ib','v']

def load():
    return FluxDvdvdv()
    
    