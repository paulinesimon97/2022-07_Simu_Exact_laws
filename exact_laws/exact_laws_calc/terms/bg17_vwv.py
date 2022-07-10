from typing import List

from .abstract_term import AbstractTerm

class Bg17Vwv(AbstractTerm):
    def __init__(self):
        pass
    
    def calc(self, values) -> (float or List[float]):
        return self.BG17_term(("vx", "vy", "vz"), ("wx", "wy", "wz"), ("vx", "vy", "vz"), datadic=values)

    def variables(self) -> List[str]:
        return ['w','v']

def load():
    return Bg17Vwv()
    
    