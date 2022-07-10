from typing import List

from .abstract_term import AbstractTerm

class FluxDvdvdv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, values) -> (float or List[float]):
        return self.flux(("vx", "vy", "vz"), ("vx", "vy", "vz"), ("vx", "vy", "vz"), datadic=values)

    def variables(self) -> List[str]:
        return ['v']

def load():
    return FluxDvdvdv()
    
    