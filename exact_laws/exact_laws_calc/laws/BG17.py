from ..abstract_law import AbstractLaw


class Bg17(AbstractLaw):
    def __init__(self):
        a = 1


def load():
    print("Hello from BG17")
    return Bg17()
