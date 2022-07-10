import logging

class Grid:
    """Classe contenant les informations nécessaires sur une grille de données
    Mère de la classe contenant les informations nécessaires sur la grille finale (grille d'échelles)
    """

    def __init__(self, N=[], L=[], c=[]):
        """Initialisation of the mother class grid"""
        self.N = N  # Echantillonnage (Nombre de points)
        self.L = L  # Largeur totale réelle
        self.c = c  # real case length
    
    def from_dict(dict):
        return Grid(N=dict['N'].astype(int),L=dict['L'],c=dict['c'])

    def check(self, name):
        """Display the principal parameter of the grid object"""
        message = f"Check Grid object {name}:"
        message += f"\n\t - N: {self.N}"
        message += f"\n\t - L: {self.L}"
        message += f"\n\t - c: {self.c}"
        logging.info(message)