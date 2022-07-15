try:
    pass
    #import mpi4py
    #mpi4py.rc.recv_mprobe = False
    #from mpi4py import MPI
except:
    pass

import logging
from datetime import datetime
import time
import numpy as np
import os

class Mpi:
    """Classe contenant les informations nécessaire à la parallelisation (wrap MPI)"""

    def __init__(self):
        """Initialisation of the class Mpi"""
        try: # case with parallelisation
            self.comm = MPI.COMM_WORLD  # MPI communicator
            self.size = self.comm.Get_size()  # MPI nb of processor
            self.rank = self.comm.Get_rank()  # MPI rank
            self.op = MPI.SUM   # gather operator
            self.type = MPI.DOUBLE
        except: # case without parallelisation
            self.comm = None
            self.size = 1
            self.rank = 0
            self.op = None
            self.type = None
        self.time_deb = datetime.now()  # date for following
        self.TAG = False  # tag for exit all processor if error
        
        # attributes that need to be modified according to the config file
        self.bufnum = 1
        self.nblayer = 1  # int(reduce(lambda x,y:x*y, N)//(128*128*128))
        
        # attributes that will be modified by counter
        self.count = [None]  # nb of processor in each group
        self.displ = [None]  # displacement: the starting rank of each group
        self.group_size = self.size  # nb of group
        self.group_rank = self.rank  # group rank
        
    def set_bufnum(self,value):
        self.bufnum = value
    
    def set_nblayer(self,value):
        self.nblayer = value
    
    def set_distrib_params(self, N):
        self.nblayer = min(self.nblayer, self.size)
        self.counter(N)
        
    def counter(self,size_cube):
        """Record count and displ attributes from the size N of the box to distribute"""
        nbmin_lines, residual_lines = divmod(size_cube[0], self.nblayer)
        nbmin_receivers, residual_receivers = divmod(self.size, self.nblayer)
        rank_receiver = 0
        list_nblines_by_layer = [] 
        list_nblines_by_receiver = [] 
        list_first_index_by_receiver = [] 
        for index_layer in range(self.nblayer): 
            nb_receivers = nbmin_receivers + 1 * bool(index_layer < residual_receivers)
            list_nblines_by_layer.append(nbmin_lines + 1 * bool(index_layer < residual_lines))
            for index_receiver in range(nb_receivers): 
                if self.rank == rank_receiver: 
                    self.group_size = nb_receivers
                    self.group_rank = index_receiver
                list_nblines_by_receiver.append(list_nblines_by_layer[index_layer])  
                list_first_index_by_receiver.append(sum(list_nblines_by_layer[:index_layer])) 
                rank_receiver += 1
        self.count = np.array(list_nblines_by_receiver)
        self.displ = np.array(list_first_index_by_receiver)
        
    def barrier(self):
        if self.size != 1:
            self.comm.Barrier()
    
    def bcast(self,object):
        if self.size != 1:
            return self.comm.bcast(object,root=0)
        else: 
            return object
        
    def reduce(self,object):
        if self.size != 1:
            total = self.comm.reduce(object, op=self.op, root=0)
        else:
            total = object
        return total
        
    # def pprint(self, *objects, rk=0):
    #     if self.rank == rk:
    #         with open(f"output_ELcalc_{self.time_deb.strftime('%d%m%Y_%H%M')}.txt", "a") as file_print:
    #             print(*objects, sep=" ", end="\n", file=file_print, flush=True)

    def check(self,name):
        """Display of the distribution's parameters recorded in the class Mpi if processor of rank 0"""
        #self.pprint(
        #    f"Data scattered along x :\n    - Nb layer : {self.Nblayer}\n    - count : {self.count}\n    - displ : {self.displ}\n"
        #)
        #self.pprint(f"Data sent via : \n    - Nb buf : {self.bufnum}\n")
        #self.pprint(f"Group carac :\n    - size : {self.group_size}\n    - rank : {self.group_rank}\n")
        message = f"Check Mpi object {name}:"
        message += f"\n\t - Nb processors: {self.size}"
        message += f"\n\t - Distributed data scattered along z:\n\t\t - Nb layer: {self.nblayer}\n\t\t - count: {self.count}\n\t\t - displ: {self.displ}"
        message += f"\n\t - Distributed data send via {self.bufnum} buffer\n"
        message += f"\n\t - Current group carac:\n\t\t - size: {self.group_size}\n\t\t - rank: {self.group_rank}\n"
        
        logging.info(message)

    # def check_time(self, state, time_deb=None):
    #     """Display if rank 0, a state tag and the current time"""
    #     if time_deb == None:
    #         time_deb = self.time_deb
    #     timeprocess = datetime.now() - self.time_deb
    #     logging.info(
    #         f"State point {state}: {time.strftime('%dj %H:%M:%S',time.gmtime(timeprocess.total_seconds()))}\n"
    #     )
        
    def distrib(self, tab, N, div=False):
        """Distribution of tab of size N to all processor according to counter distribution's parametters and if or not a derivative is expected
        Use check to display the parametters
        Return the fraction of tab in each processor"""
        if self.count[0] == None:
            self.set_distrib_params(N)
            self.check('self')
        if self.size == 1:
            return tab  # Cas sans parallèlisation
        else:
            for b in range(self.bufnum):
                if self.rank == 0:
                    if div == False:  # découpage du cube à envoyer suivant la direction Z (cas sans divergence)
                        sendbuf = [
                            tab[:, :, self.displ[r] : self.displ[r] + self.count[r]]
                            if (r < self.size / self.bufnum * (b + 1) and r >= self.size / self.bufnum * b)
                            else 0
                            for r in range(self.size)
                        ]
                    else:  # découpage du cube à envoyer suivant la direction Z (cas avec divergence)
                        sendbuf = [
                            np.transpose(
                                np.concatenate(
                                    (
                                        np.transpose(tab[:, :, self.displ[r] : self.displ[r] + self.count[r] + 1]),
                                        [np.transpose(tab[:, :, self.displ[r] - 1])],
                                    ),
                                    axis=0,
                                )
                            )
                            if (r < self.size / self.bufnum * (b + 1) and r >= self.size / self.bufnum * b)
                            else 0
                            for r in range(self.size)
                        ]  # au dessus de la couche de largeur count[r] on ajoute le plan count[r]+1 puis le plan displ[r]-1
                else:
                    sendbuf = None
                recvbuf = self.comm.scatter(sendbuf, root=0)
                if self.rank < self.size / self.bufnum * (b + 1) and self.rank >= self.size / self.bufnum * b:
                    recv = recvbuf
                self.barrier()
            return recv
    
    def configure_log(self,name):
        if name == '':
            folder = f"log_{self.time_deb.strftime('%d%m%Y_%H%M%S')}"
        else:
            folder = f"log_{name}_{self.time_deb.strftime('%d%m%Y_%H%M%S')}"
        if self.rank == 0: os.mkdir(folder)
        self.barrier()
        filename = f"{folder}/{folder[4:]}_rank{self.rank}.log"
        logging.basicConfig(filename=filename, 
                        level=logging.DEBUG, 
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        )
