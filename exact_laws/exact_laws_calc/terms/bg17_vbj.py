from typing import List

from .abstract_term import AbstractTerm

class Bg17Vbj(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, values) -> (float or List[float]):
        return self.BG17_term(("vx", "vy", "vz"), ("Ibx", "Iby", "Ibz"), ("Ijx", "Ijy", "Ijz"), datadic=values)

    def variables(self) -> List[str]:
        return ['Ij','Ib','v']

def load():
    return Bg17Vbj()
    
    