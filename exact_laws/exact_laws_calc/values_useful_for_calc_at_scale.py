import logging
import numpy as np

class ValuesUsefulForCalcAtScale:
    """Classe de calcul
    Contient un dictionnaires des données d'origine et des données translatées ainsi que les fonctions de calcul
    """
    def __init__(self, data_origin, mpi):
        """distribution fragments data non décallés"""
        self.keys = []
        if mpi.rank == 0:
            self.keys = [k for k in data_origin.quantities.keys()]
        if mpi.size != 1:
            self.keys = mpi.bcast(self.keys)
        self.datadic = {}
        for k in self.keys:
            if mpi.rank == 0:
                tab = data_origin.quantities[k]
            else:
                tab = 0
            self.datadic[k] = mpi.distrib(tab, data_origin.grid.N)

    def check(self, mpi, rank=0):
        message = f"Check ValuesUsefulForCalcAtScale object at rank {rank}:"
        message += f"\n\t - keys: {self.keys}"
        message += f"\n\t - datadic keys: {self.datadic.keys()}"
        logging.info(message)

    def set_data_dir0(self, data_origin, vector, mpi):
        for k in self.keys:
            if mpi.rank == 0:
                tab = np.roll(data_origin.quantities[k], -np.array(vector), axis=(0, 1, 2))
            else:
                tab = 0
            self.datadic[k + "0"] = mpi.distrib(tab, data_origin.grid.N, div=True)

    def set_data_prim(self, vector):
        shape = np.shape(self.datadic[self.keys[0]])
        for k in self.keys:
            self.datadic[k + "P"] = np.roll(self.datadic[k + "0"], -np.array(vector), axis=(0, 1, 2))[
                : shape[0], : shape[1], : shape[2]
            ]