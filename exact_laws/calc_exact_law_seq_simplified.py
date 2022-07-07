# # **SEQU Calculation Exact Law**

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Pauline SIMON
# dernière modification : 26/06/2022
version = "26/06/2022"

# Modules
from contextlib import redirect_stdout
from builtins import *
import numpy as np
import numexpr as ne
from functools import reduce
import pandas as pd
from datetime import datetime
import time
import random

import sys
import h5py as h5
import pickle as pkl
import os

import cProfile, sys

import mpi4py

mpi4py.rc.recv_mprobe = False
try:
    from mpi4py import MPI
except:
    pass


class Math_Tools:
    def cdiff(tab, length_case=1, dirr=0, precision=4, period=True, point=False):
        """This function does a central discrete derivation at ordre 1 with precision 2 or 4.
        Args:
            tab: array
                Data of only une quantity on which is applied the derivation.
            length_case: float (default = 1)
                Length of a case of data in the axis of the derivation.
            dirr: int (default = 0)
                Axis of the derivation.
                Cautious: important if 'tab' has more than one dimension.
            precision: int (default = 4)
                Precision of the discrete methode, 2 or 4.
                Cautious: if 2, the function need one-step-left and right values.
                Cautious: if 4, the function need two-step-left and right values.
            period: bool (default = True)
                Periodical edge effect.
                Cautious: if False a right or left discrete method is used for the edges.
        Return: numpy.array
        """
        if len(np.shape(tab)) == 1:
            dirr = 0
        if length_case == 0:
            return np.zeros(np.shape(tab))
        tab = np.array(tab)

        if precision == 4:
            if point == False:
                result = (
                    np.roll(tab, 2, axis=int(dirr))
                    - 8 * np.roll(tab, 1, axis=int(dirr))
                    + 8 * np.roll(tab, -1, axis=int(dirr))
                    - np.roll(tab, -2, axis=int(dirr))
                ) / (12 * length_case)
                if period == False:
                    if dirr == 0:
                        result[0] = (tab[1] - tab[0]) / length_case  # diff droite
                        result[1] = (tab[2] - tab[0]) / (2 * length_case)  # diff centrée ordre 2
                        result[-2] = (tab[-1] - tab[-3]) / (2 * length_case)  # diff centrée ordre 2
                        result[-1] = (tab[-1] - tab[-2]) / length_case  # diff gauche
                    elif dirr == 1:
                        result[:, 0] = (tab[:, 1] - tab[:, 0]) / length_case
                        result[:, 1] = (tab[:, 2] - tab[:, 0]) / (2 * length_case)
                        result[:, -2] = (tab[:, -1] - tab[:, -3]) / (2 * length_case)
                        result[:, -1] = (tab[:, -1] - tab[:, -2]) / length_case
                    elif dirr == 2:
                        result[:, :, 0] = (tab[:, :, 1] - tab[:, :, 0]) / length_case
                        result[:, :, 1] = (tab[:, :, 2] - tab[:, :, 0]) / (2 * length_case)
                        result[:, :, -2] = (tab[:, :, -1] - tab[:, :, -3]) / (2 * length_case)
                        result[:, :, -1] = (tab[:, :, -1] - tab[:, :, -2]) / length_case
            else:
                result = (tab[0] + 8 * tab[-2] - 8 * tab[1] - tab[-1]) / (12 * length_case)  # diff locale ordre 4

        elif precision == 2:
            if point == False:
                result = (np.roll(tab, -1, axis=int(dirr)) - np.roll(tab, 1, axis=int(dirr))) / (2 * length_case)
                if period == False:
                    if dirr == 0:
                        result[0] = (tab[1] - tab[0]) / length_case  # diff droite
                        result[-1] = (tab[-1] - tab[-2]) / length_case  # diff gauche
                    elif dirr == 1:
                        result[:, 0] = (tab[:, 1] - tab[:, 0]) / length_case
                        result[:, -1] = (tab[:, -1] - tab[:, -2]) / length_case
                    elif dirr == 2:
                        result[:, :, 0] = (tab[:, :, 1] - tab[:, :, 0]) / length_case
                        result[:, :, -1] = (tab[:, :, -1] - tab[:, :, -2]) / length_case
            else:
                result = (tab[-1] - tab[0]) / (2 * length_case)
            # diff locale ordre 2
        return result


class Mpi:
    """Classe contenant les informations nécessaire à la parallélisation (wrap MPI)"""

    def __init__(self):
        """Initialisation of the class Mpi"""
        try:
            self.comm = MPI.COMM_WORLD  # MPI communicator
        except:
            self.comm = None
        try:
            self.size = self.comm.Get_size()  # MPI nb of processor
        except:
            self.size = 1
        try:
            self.rank = self.comm.Get_rank()  # MPI rank
        except:
            self.rank = 0
        try:
            self.op = MPI.SUM  # gather operator
        except:
            self.op = 0
        try:
            self.type = MPI.DOUBLE
        except:
            self.type = 0
        self.time_deb = datetime.today()  # date for following
        self.TAG = False  # tag for exit all processor if error
        self.count = [None]  # nb of processor in each group
        self.displ = [None]  # displacement: the starting rank of each group
        self.group_size = self.size  # nb of group
        self.group_rank = self.rank  # group rank
        self.bufnum = 2
        self.Nblayer = 8  # int(reduce(lambda x,y:x*y, N)//(128*128*128))

    def pprint(self, *objects, rk=0):
        if self.rank == rk:
            with open(f"output_ELcalc_{self.time_deb.strftime('%d%m%Y_%H%M')}.txt", "a") as file_print:
                print(*objects, sep=" ", end="\n", file=file_print, flush=True)

    def check(self):
        """Display of the distribution's parameters recorded in the class Mpi if processor of rank 0"""
        self.pprint(
            f"Data scattered along x :\n    - Nb layer : {self.Nblayer}\n    - count : {self.count}\n    - displ : {self.displ}\n"
        )
        self.pprint(f"Data sent via : \n    - Nb buf : {self.bufnum}\n")
        self.pprint(f"Group carac :\n    - size : {self.group_size}\n    - rank : {self.group_rank}\n")

    def check_time(self, state, time_deb=None):
        """Display if rank 0, a state tag and the current time"""
        if time_deb == None:
            time_deb = self.time_deb
        if state != "INIT" and state != "END":
            timeprocess = datetime.today() - self.time_deb
            self.pprint(
                f"State point {state}: {time.strftime('%dj %H:%M:%S',time.gmtime(timeprocess.total_seconds()))}\n"
            )
        elif state == "INIT":
            self.pprint(f"Beginning: {self.time_deb.strftime('%d/%m/%Y %H:%M:%S')}\n")
        elif state == "END":
            self.pprint(f"End: {datetime.today().strftime('%d/%m/%Y %H:%M:%S')}\n")

    def counter(self, N):
        """Record count and displ attributes from the size N of the box to distribute"""
        Nb_layer = min(self.Nblayer, self.size)
        self.Nblayer = Nb_layer
        ave, res = divmod(N[0], Nb_layer)  # ave : min(nb lignes/layer), res : nb layer tel que nb ligne = ave+1
        Q, R = divmod(self.size, Nb_layer)  # Q : min(nb proc/layer), R : nb layer transmise à Q+1 proc
        r = 0  # index du processeur (sera indenté par la suite)
        c = []  # liste des nombre de lignes contenue par une couche
        count = []  # count: nb lignes par processeur
        displ = []  # displacement: the starting index de la couche associée à chaque processeur
        for p in range(Nb_layer):  # distribution des couches
            if p < R:
                n = Q + 1  # nombre de processeurs contenant la couche p (cas "layer transmise à Q+1 proc")
            else:
                n = Q  # nombre de processeurs contenant la couche p (cas "layer transmise à Q proc")
            if p < res:
                c.append(
                    ave + 1
                )  # ajout à la liste c du nombre de lignes de la couche p (cas "layer tel que nb ligne = ave+1")
            else:
                c.append(
                    ave
                )  # ajout à la liste c du nombre de lignes de la couche p (cas "layer tel que nb ligne = ave")
            for i in range(n):  # distribution aux processeurs du groupe
                if self.rank == r:  # indication au processeur de la taille du groupe et de son rang dans le groupe
                    self.group_size = n
                    self.group_rank = i
                count.append(c[p])  # ajout à count du nombre de ligne contenue par le processeur r
                displ.append(sum(c[:p]))  # ajout à displ du starting index de la couche contenue par le processeur r
                r += 1
        self.count = np.array(count)
        self.displ = np.array(displ)

    def distrib(self, tab, N, div=False):
        """Distribution of tab of size N to all processor according to counter distribution's parametters and if or not a derivative is expected
        Use check to display the parametters
        Return the fraction of tab in each processor"""
        if self.count[0] == None:
            self.counter(N)
            self.check()
        if self.size == 1:
            return tab  # Cas sans parallèlisation
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
            self.comm.Barrier()
        return recv


class Grid:
    """Classe contenant les informations nécessaires sur une grille de données
    Mère de la classe contenant les informations nécessaires sur la grille finale (grille d'échelles)
    """

    def __init__(self, N, L, c):
        """Initialisation of the mother class grid"""
        self.N = N  # Echantillonnage (Nombre de points)
        self.L = L  # Largeur totale réelle
        self.c = c  # real case length

    def check(self, mpi):
        """Display the principal parameter of the grid object"""
        mpi.pprint(f"Grid (x,y,z) :\n   - N : {self.N}\n   - L : {self.L}\n   - c : {self.c}\n")


class Grid_scale_logcyl(Grid):
    """Classe contenant les informations nécessaires sur la grille finale (grille d'échelles)"""

    def __init__(self, N, L, c, Nmax_scale, Nmax_list, kind="cls"):
        """Initialisation of class grid_scale_logcyl
        directions : lperp (norme dans le plan polaire),z (coupe parallèle)
        N : list of maximum number of points in each dimension of the scale grid (argument = possible, attribut final = final)
        indexes selected according to point_min and N, then N is reevaluated"""
        grid = {}

        # liste des échelles parallèles (indexes des coupes parallèles)
        grid["lz"] = np.unique(np.logspace(0, np.log10(N[2]), Nmax_scale, endpoint=True, dtype=int))
        grid["lz"] = np.array(np.append([0], grid["lz"]))
        N_par = np.shape((grid["lz"]))[0]

        # liste des échelles perp (norme polaire)
        N_perp = np.min(N[0:2])
        grid["lperp"] = np.unique(np.logspace(0, np.log10(N_perp), Nmax_scale, endpoint=True, dtype=int))
        grid["lperp"] = np.array(np.append([0], grid["lperp"]))
        N_perp = np.shape((grid["lperp"]))[0]

        # init liste des points présent sur un même arc de rayon lperp +/- une marge
        grid["listperp"] = [[] for r in range(N_perp)]

        # init liste des écarts entre les points d'un arc et l'arc
        grid["listnorm"] = [[] for r in range(N_perp)]

        # init nombre de points présent sur un même arc
        grid["count"] = np.zeros((N_perp))

        # precision : marge maximale acceptée entre un point et l'arc associé
        precision = [
            np.min(((grid["lperp"][r + 1] - grid["lperp"][r]) / 2, (grid["lperp"][r] - grid["lperp"][r - 1]) / 2))
            for r in range(1, N_perp - 1)
        ]
        precision = np.append([(grid["lperp"][1] - grid["lperp"][0]) / 2], precision)  # pour le premier point
        precision = np.append(precision, [(grid["lperp"][-1] - grid["lperp"][-2]) / 2])  # pour le dernier point

        # distribution des points cartésiens suivant les arcs
        limX = np.min((grid["lperp"][-1], N[0]))  # limite d'intéret en x
        limY = np.min((grid["lperp"][-1], N[1]))  # limite d'intéret en y
        for x in range(-limX, limX + 1):  # boucle sur l'ensemble des points d'échelle d'intéret (y imbriqué dans x)
            for y in range(-limY, limY + 1):
                n = np.sqrt(x * x + y * y)  # norme du point
                for r in range(N_perp):  # boucle sur l'ensemble des normes sélectionnées
                    if n >= grid["lperp"][r] - precision[r] and n <= grid["lperp"][r] + precision[r]:
                        grid["listperp"][r].append(
                            [x, y]
                        )  # distribution des points en fonction de leur présence près d'un arc de rayon lperp
                        grid["listnorm"][r].append(np.abs(grid["lperp"][r] - n))
                        grid["count"][r] += 1
                        continue

        # réduction aléatoire (rdm) ou au plus près (cls) du nombre de points près d'un arc lperp à Nmax_list points si supérieur
        for r in range(N_perp):
            if grid["count"][r] > Nmax_list:
                if kind == "cls":
                    grid["listperp"][r] = [x for _, x in sorted(zip(grid["listnorm"][r], grid["listperp"][r]))][
                        :Nmax_list
                    ]
                elif kind == "rdm":
                    grid["listperp"][r] = random.sample(grid["listperp"][r], Nmax_list)
                grid["count"][r] = Nmax_list

        # creation de la grille
        Grid.__init__(self, np.array([N_par, N_perp, np.max(grid["count"])], dtype=int), L, c)
        self.grid = grid
        # N tel que [nombre de coupes, nombre de rayons dans le plan polaire, nombre maximal de direction dans le plan polaire par rayon]


class OriginalDataset:
    """Classe des données"""

    def __init__(self, param, datadic, scale):
        """Initialisation of the mother class data"""
        self.param = param
        self.datadic = datadic  # dictionary
        self.scale = scale  # grid

    def check(self, mpi):
        """display all quantities' mean"""
        if mpi.rank == 0:
            text = f"Data param:\n"
            for k in self.param.keys():
                text += f"   - {k} = {self.param[k]}\n"
            text += f"Data's quantities: \n"
            for k in self.datadic.keys():
                tab = self.datadic[k]
                text += f"   - {k}\n"  # = {np.mean(np.sort(tab.copy().reshape(np.product(tab.shape))))}")
            mpi.pprint(text)
        self.scale.check(mpi)
        mpi.pprint("")

    def data_read(file_name):
        """creat a datadic from data in file_name"""
        data = {}
        param = {}
        law = {}
        with h5.File(file_name, "r") as f:
            for k in f.keys():
                if not "param" in k:
                    data[k] = np.ascontiguousarray(f[k])
            param["L"] = np.array(f["param"]["L"])
            param["N"] = np.array(f["param"]["N"])
            param["c"] = np.array(f["param"]["c"])
            param["cycle"] = str(np.array(f["param"]["cycle"]))
            param["di"] = float(np.array(f["param"]["di"]))
            param["bin"] = float(np.array(f["param"]["bin"]))
            param["rho_mean"] = 1  # np.mean(np.sort(data['rho'].flatten()))
            for p in [
                "BG17",
                "BG17Hall",
                "SS22C",
                "SS22CGyr",
                "SS22CHall",
                "SS22CIso",
                "SS22I",
                "SS22IGyr",
                "SS22IHall",
                "SS21C",
                "SS21CIso",
            ]:
                law[p] = bool(np.array(f["param"][p]))
        return data, param, law


class ResultExactLaw():
    """Classe des données finales
    Contient : un dictionaire de paramètre param, un dictionaire avec les données datadic et des information sur la grille des données scale
    """

    def __init__(self, param, data_origin, Nmax_scale, Nmax_list):
        # creation de la grille logarithmique
        Ns = data_origin.param["N"] / 2  # the maximum number of points available or wanted between [1,1,1] et N/2
        Ls = data_origin.param["L"] / 2  # maximal gap possible between two points
        c = data_origin.param["c"]  # case length for local derivation and minimal gap posible between two points
        scale = Grid_scale_logcyl(Ns.astype(int), Ls, c, Nmax_scale, Nmax_list, param["kind"])

        # obtention de la liste des termes voulus, ainsi que des outils pour les calculer

        terms, fonctions, arguments, coeff = ResultExactLaw.get_terms(param, data_origin.param)

        # initialisation des cubes de données (datadic)
        output_shape = scale.N  # taille pour termes scalaires
        output_shape_flux = np.append(output_shape, 3)  # taille pour termes vectoriels
        output_shape_div = np.append(output_shape_flux, 2)  # taille pour termes finalement dérivés
        output_dict = {}
        for i, t in enumerate(terms):
            if t.startswith("flux"):
                output_dict[t] = np.zeros(output_shape_flux, dtype=np.float64)
                output_dict[f"div_{t}"] = np.zeros(output_shape, dtype=np.float64)
                output_dict[f"term_div_{t}"] = np.zeros(output_shape_div, dtype=np.float64)
            else:
                output_dict[t] = np.zeros(output_shape, dtype=np.float64)

        # initialisation des attributs
        self.param = param
        self.datadic = output_dict  # dictionary
        self.scale = scale
        self.coeff = coeff
        self.funcdic = fonctions
        self.argdic = arguments
        self.state = 0  # int indiquant l'état de remplissage des cubes de données

    def get_terms(law, param_origin):
        terms, funct, args, coeff = [], {}, {}, {}
        if law["BG17"]:
            terms, funct, args, coeff = ResultExactLaw.get_terms_BG17(terms, funct, args, coeff, param_origin)
        if law["SS22I"]:
            terms, funct, args, coeff = ResultExactLaw.get_terms_SS22I(terms, funct, args, coeff, param_origin)
        if law["SS22IGyr"]:
            terms, funct, args, coeff = ResultExactLaw.get_terms_SS22IGyr(terms, funct, args, coeff, param_origin)
        return terms, funct, args, coeff

    def get_terms_BG17(terms, funct, args, coeff, param_origin, helm=False):
        # term 'BG17_vwv'
        if helm == False:
            t = "BG17_vwv"
        else:
            t = "BG17_vwv_helm"
        terms.append(t)
        funct[t] = FonctionTermAtScale.BG17_term
        if helm == False:
            args[t] = [("vx", "vy", "vz"), ("wx", "wy", "wz"), ("vx", "vy", "vz")]
        else:
            args[t] = [("Ivx", "Ivy", "Ivz"), ("Iwx", "Iwy", "Iwz"), ("Ivx", "Ivy", "Ivz")]
        coeff[t] = param_origin["rho_mean"] / 2

        # term 'BG17_jbv'
        if helm == False:
            t = "BG17_jbv"
        else:
            t = "BG17_jbv_helm"
        terms.append(t)
        funct[t] = FonctionTermAtScale.BG17_term
        if helm == False:
            args[t] = [("Ijx", "Ijy", "Ijz"), ("Ibx", "Iby", "Ibz"), ("vx", "vy", "vz")]
        else:
            args[t] = [("Ijx", "Ijy", "Ijz"), ("Ibx", "Iby", "Ibz"), ("Ivx", "Ivy", "Ivz")]
        coeff[t] = param_origin["rho_mean"] / 2

        # term 'BG17_vbj'
        if helm == False:
            t = "BG17_vbj"
        else:
            t = "BG17_vbj_helm"
        terms.append(t)
        funct[t] = FonctionTermAtScale.BG17_term
        if helm == False:
            args[t] = [("vx", "vy", "vz"), ("Ibx", "Iby", "Ibz"), ("Ijx", "Ijy", "Ijz")]
        else:
            args[t] = [("Ivx", "Ivy", "Ivz"), ("Ibx", "Iby", "Ibz"), ("Ijx", "Ijy", "Ijz")]
        coeff[t] = param_origin["rho_mean"] / 2
        return terms, funct, args, coeff

    def get_terms_SS22I(terms, funct, args, coeff, param_origin, helm=False):

        # term 'flux_dvdvdv'
        t = "flux_dvdvdv"
        if helm == True:
            t = t + "_helm"
        terms.append(t)
        funct[t] = FonctionTermAtScale.flux
        if helm == False:
            args[t] = [("vx", "vy", "vz"), ("vx", "vy", "vz"), ("vx", "vy", "vz")]
        else:
            args[t] = [("Ivx", "Ivy", "Ivz"), ("Ivx", "Ivy", "Ivz"), ("Ivx", "Ivy", "Ivz")]
        coeff[f"div_{t}"] = -param_origin["rho_mean"] / 4

        # term 'flux_dbdbdv'
        t = "flux_dbdbdv"
        if helm == True:
            t = t + "_helm"
        terms.append(t)
        funct[t] = FonctionTermAtScale.flux
        if helm == False:
            args[t] = [("Ibx", "Iby", "Ibz"), ("Ibx", "Iby", "Ibz"), ("vx", "vy", "vz")]
        else:
            args[t] = [("Ibx", "Iby", "Ibz"), ("Ibx", "Iby", "Ibz"), ("Ivx", "Ivy", "Ivz")]
        coeff[f"div_{t}"] = -param_origin["rho_mean"] / 4

        # term 'flux_dvdbdb'
        t = "flux_dvdbdb"
        if helm == True:
            t = t + "_helm"
        terms.append(t)
        funct[t] = FonctionTermAtScale.flux
        if helm == False:
            args[t] = [("vx", "vy", "vz"), ("Ibx", "Iby", "Ibz"), ("Ibx", "Iby", "Ibz")]
        else:
            args[t] = [("Ivx", "Ivy", "Ivz"), ("Ibx", "Iby", "Ibz"), ("Ibx", "Iby", "Ibz")]
        coeff[f"div_{t}"] = param_origin["rho_mean"] / 2

        return terms, funct, args, coeff

    def get_terms_SS22IGyr(terms, funct, args, coeff, param_origin, helm=False, meth=1):

        # term 'source_dpan'
        t = "source_dpan"
        if helm == True:
            t = t + "_helm"
        if meth != 1:
            t = t + "_2"
        terms.append(t)
        funct[t] = FonctionTermAtScale.source_dp
        args[t] = [
            "NA",
        ]
        if helm == True:
            args[t].append("I")
        else:
            args[t].append("")
        if meth != 1:
            args[t].append("2")
        coeff[t] = 1 / 4

        return terms, funct, args, coeff

    def fill_result_at_scale(self, indices, values, div=False):
        """Remplissage des cubes au point d'indices 'indices' à partir des dictionnaires funcdic et argdic
        si div = False : remplissage des cubes flux et sources
        si div = True : remplissage des cubes term_div
        """
        if div == False:
            for t in self.funcdic.keys():
                self.datadic[t][indices[0], indices[1], indices[2]] = self.funcdic[t](values, *self.argdic[t])
        else:
            for t in self.funcdic.keys():
                if t.startswith("flux"):
                    self.datadic[f"term_div_{t}"][
                        indices[0], indices[1], indices[2], indices[3], indices[4]
                    ] = self.funcdic[t](values, *self.argdic[t])[indices[3]]

    def record_to_h5file(self, filename):
        with h5.File(filename, "w") as file:
            file.create_group("param")
            for k in self.param.keys():
                file["param"].create_dataset(k, data=self.param[k])
            file.create_group("coeff")
            for k in self.coeff.keys():
                file["coeff"].create_dataset(k, data=self.coeff[k])
            file.create_group("data")
            for k in self.datadic.keys():
                file["data"].create_dataset(k, data=self.datadic[k])
            file.create_group("grid")
            for k in self.scale.grid.keys():
                if "listperp" in k:
                    n = len(max(self.scale.grid[k], key=len))
                    lst_2 = [x + [[np.nan, np.nan]] * (n - len(x)) for x in self.scale.grid[k]]
                    file["grid"].create_dataset(k, data=lst_2)
                elif "listnorm" in k:
                    n = len(max(self.scale.grid[k], key=len))
                    lst_2 = [x + [np.nan] * (n - len(x)) for x in self.scale.grid[k]]
                    file["grid"].create_dataset(k, data=lst_2)
                else:
                    file["grid"].create_dataset(k, data=self.scale.grid[k])


class ValuesUsefulForCalcAtScale:
    """Classe de calcul
    Contient un dictionnaires des données d'origine et des données translatées ainsi que les fonctions de calcul
    """

    def __init__(self, data_origin, mpi):
        """distribution fragments data non décallés"""
        self.keys = []
        if mpi.rank == 0:
            self.keys = [k for k in data_origin.datadic.keys()]
        if mpi.size != 1:
            self.keys = mpi.comm.bcast(self.keys, root=0)
        self.datadic = {}
        for k in self.keys:
            if mpi.rank == 0:
                tab = data_origin.datadic[k]
            else:
                tab = 0
            self.datadic[k] = mpi.distrib(tab, data_origin.param["N"])

    def check(self, mpi, rank=0):
        mpi.pprint("Check ValuesUsefulForCalcAtScale object at rank ", rank, rk=rank)
        mpi.pprint("keys: ", self.keys)
        mpi.pprint("datadic keys: ", self.datadic.keys())

    def set_data_dir0(self, data_origin, vector, mpi):
        for k in self.keys:
            if mpi.rank == 0:
                tab = np.roll(data_origin.datadic[k], -np.array(vector), axis=(0, 1, 2))
            else:
                tab = 0
            self.datadic[k + "0"] = mpi.distrib(tab, data_origin.param["N"], div=True)

    def set_data_prim(self, vector):
        shape = np.shape(self.datadic[self.keys[0]])
        for k in self.keys:
            self.datadic[k + "P"] = np.roll(self.datadic[k + "0"], -np.array(vector), axis=(0, 1, 2))[
                : shape[0], : shape[1], : shape[2]
            ]

class FonctionTermAtScale:
    def flux_gen(datadic, data_1, data_2, data_3, data_4=None):
        """general expression for flux of the form (data_1 . data_2) data_3
        ex for K41 : self.flux_gen(('vx','vy','vz'),('vx','vy','vz'),('vx','vy','vz'))
        need : data_1, data_2, data_3 and conjugates in local_dict.keys()
        return an iterator on the flux components summed on all the points of the original grid"""

        if data_4 == None:  # calcul flux term with 3 quantities (ex: dv.dvdv)
            scalar_product = ""
            for i in range(len(data_1)):
                scalar_product += f" ({data_1[i]}P - {data_1[i]}) * ({data_2[i]}P - {data_2[i]}) +"
            for j in range(len(data_3)):
                tab = ne.evaluate(
                    f"   ({scalar_product[:-1]}) * ({data_3[j]}P - {data_3[j]}) ".lstrip(),
                    local_dict=datadic,
                )
                yield np.sum(tab)  # np.sum(np.sort(tab.flatten()))

        elif data_4 == "rho":  # calcul compressible flux term  (ex: drv.dvdv)
            scalar_product = ""
            for i in range(len(data_1)):
                scalar_product += (
                    f" ({data_4}P * {data_1[i]}P      "
                    f"- {data_4}  * {data_1[i]})      "
                    f"* ({data_2[i]}P - {data_2[i]}) +"
                )
            for i in range(len(data_3)):
                tab = ne.evaluate(
                    f"   ({scalar_product[:-1]}) * ({data_3[i]}P - {data_3[i]})".lstrip(),
                    local_dict=datadic,
                )
                yield np.sum(tab)  # np.sum(np.sort(tab.flatten()))

        elif data_4 == "pan":
            scalar_productP = ""
            scalar_product = ""
            for i in range(len(data_3)):
                scalar_productP += f"{data_2[i]}P * ({data_3[i]}P - {data_3[i]}) +"
                scalar_product += f"{data_2[i]}  * ({data_3[i]}P - {data_3[i]}) +"
            for i in range(len(data_2)):
                tab = ne.evaluate(
                    f"({data_1[0]}P - {data_1[0]})                 "
                    f" * ((pparP - pperpP) / pmP * {data_2[i]}P    "
                    f"   * {scalar_productP[:-1]}                  "
                    f"  - (ppar  - pperp)  / pm  * {data_2[i]}     "
                    f"   * {scalar_product[:-1]})                  ".lstrip(),
                    local_dict=datadic,
                )
                yield np.sum(tab)  # np.sum(np.sort(tab.flatten()))

    def flux(datadic, data_1, data_2, data_3, data_4=None):
        """general expression for flux of the form (data_1 . data_2) data_3
        ex for K41 : self.flux(('vx','vy','vz'),('vx','vy','vz'),('vx','vy','vz'))
        need : data_1, data_2, data_3 and conjugates in local_dict.keys()
        return a list of the flux components summed on all the points of the original grid"""
        return np.array(list(FonctionTermAtScale.flux_gen(datadic, data_1, data_2, data_3, data_4)))

    def source_dp(datadic, data_p, helm="", meth=1):

        exprP = f"(IpparP - IpperpP) / (IpmP)"
        expr = f"- (Ippar - Ipperp) / (Ipm)"

        if meth == 1:
            pdualPP, pdualP1, pdual1P, pdual11 = f"", f"", f"", f""
            for i in ("x", "y", "z"):
                for j in ("x", "y", "z"):
                    pdualPP += f"Ib{i}P * Ib{j}P * {helm}d{i}v{j}P +"
                    pdualP1 += f"Ib{i}P * Ib{j}P * {helm}d{i}v{j} +"
                    pdual1P += f"Ib{i}  * Ib{j}  * {helm}d{i}v{j}P +"
                    pdual11 += f"Ib{i}  * Ib{j}  * {helm}d{i}v{j} +"
            tab = ne.evaluate(f"{exprP} * (({pdualPP[:-1]}) - ({pdualP1[:-1]}))".lstrip(), local_dict=datadic)
            out = np.sum(tab)
            tab = ne.evaluate(f"{expr} * (({pdual1P[:-1]}) - ({pdual11[:-1]}))".lstrip(), local_dict=datadic)
            out = out + np.sum(tab)

        else:
            dualP, dual1 = f"", f""
            for i in ("x", "y", "z"):
                for j in ("x", "y", "z"):
                    dualP += f"Ib{i}P * Ib{j}P * ({helm}d{i}v{j}P - {helm}d{i}v{j})+"
                    dual1 += f"Ib{i}  * Ib{j}  * ({helm}d{i}v{j}P - {helm}d{i}v{j})+"
            tab = ne.evaluate(f"{exprP} * ({dualP[:-1]})".lstrip(), local_dict=datadic)
            out = np.sum(tab)
            tab = ne.evaluate(f"{expr} * ({dual1[:-1]})".lstrip(), local_dict=datadic)
            out = out + np.sum(tab)

        return out

    def BG17_term(datadic, data_1, data_2, data_3):
        """general expression for BG17 terms in of the form delta(data_1 x data_2).delta(data_3)
        need : data_1, data_2, data_3 and conjugates in local_dict.keys()
        return the term summed on all the points of the original grid"""
        d0 = (
            f"  {data_1[1]}P * {data_2[2]}P - {data_1[2]}P * {data_2[1]}P "
            f"- {data_1[1]}  * {data_2[2]}  + {data_1[2]}  * {data_2[1]}  "
        )
        d1 = (
            f"  {data_1[2]}P * {data_2[0]}P - {data_1[0]}P * {data_2[2]}P "
            f"- {data_1[2]}  * {data_2[0]}  + {data_1[0]}  * {data_2[2]}  "
        )
        d2 = (
            f"  {data_1[0]}P * {data_2[1]}P - {data_1[1]}P * {data_2[0]}P "
            f"- {data_1[0]}  * {data_2[1]}  + {data_1[1]}  * {data_2[0]}  "
        )
        tab = ne.evaluate(
            f"  ({d0}) * ({data_3[0]}P - {data_3[0]}) "
            f"+ ({d1}) * ({data_3[1]}P - {data_3[1]}) "
            f"+ ({d2}) * ({data_3[2]}P - {data_3[2]}) ".lstrip(),
            local_dict=datadic,
        )
        return np.sum(tab)  # np.sum(np.sort(tab.flatten()))


# MAIN __________________________________________________________________________________________________________________________________________
if __name__ == "__main__":
    # ## Input et initialisation
    mpi = Mpi()
    folder = "./"
    if mpi.rank == 0:
        print(f"Process information recorded in output_ELcalc_{mpi.time_deb.strftime('%d%m%Y_%H%M')}.txt")
    # mpi.check()
    # mpi.comm.Barrier()
    # mpi.pprint("Thread ready:", mpi.rank, rk=mpi.rank)
    # mpi.comm.Barrier()
    mpi.pprint(f"Run of calc_exact_law_seq.py version {version} the {mpi.time_deb.strftime('%d/%m/%Y at %H:%M')}.\n")
    if mpi.rank == 0:
        os.mkdir(f"./save_{mpi.time_deb.strftime('%d%m%Y_%H%M')}")  # creation of a recording folder
    save_folder = f"./save_{mpi.time_deb.strftime('%d%m%Y_%H%M')}/"

    # File management : read input txt file information
    inputdic = {}
    with open(folder + "input_calc.txt", encoding="utf-8") as entree:
        for line in entree:
            value = line.split()
            if len(value) >= 2:
                inputdic[value[0]] = value[1]

    mpi.Nblayer = int(inputdic["Nblayer"])
    mpi.bufnum = int(inputdic["Nbbuf"])

    # Attribut data_origin
    datafile = inputdic["folder_data"] + inputdic["name_data"] + ".h5"
    mpi.pprint(f"Original data file: {datafile }")

    if "save" in inputdic:
        mpi.check_time(f"Download original data from save folder {inputdic['save']} INIT")
        if mpi.rank == 0:
            file_record = f"{inputdic['save']}data_origin_master.pkl"
        else:
            file_record = f"{inputdic['save']}data_origin_slave.pkl"
        with open(file_record, "rb") as recordfile:
            data_origin = pkl.load(recordfile)
        if mpi.size != 1:
            mpi.comm.Barrier()
        mpi.check_time(f"Download original data from save folder {inputdic['save']} END")

    else:
        mpi.check_time("Initialisation original data INIT")
        data, scale, param, law = 0, 0, 0, 0
        if mpi.rank == 0:
            data, param, law = OriginalDataset.data_read(datafile)
        if mpi.size != 1:
            param = mpi.comm.bcast(param, root=0)
            law = mpi.comm.bcast(law, root=0)
        scale = Grid(param["N"].astype(int), param["L"], param["c"])
        if mpi.size != 1:
            mpi.comm.Barrier()
        data_origin = OriginalDataset(param, data, scale)
        del (data, scale, param)
        mpi.check_time("Initialisation original data END")

    data_origin.check(mpi)

    mpi.check_time("Record original data INIT")
    if mpi.rank == 0:
        file_record = f"{save_folder}data_origin_master.pkl"
    if mpi.rank == 1:
        file_record = f"{save_folder}data_origin_slave.pkl"
    if mpi.rank == 0 or mpi.rank == 1:
        with open(file_record, "wb") as recordfile:
            pkl.dump(data_origin, recordfile)
    mpi.check_time("Record original data END")
    if mpi.size != 1:
        mpi.comm.Barrier()

    # Attribut Output

    if "save" in inputdic:
        mpi.check_time(f"Download output data from save folder {inputdic['save']} INIT")
        with open(f"{inputdic['save']}data_output_rk{mpi.rank}.pkl", "rb") as recordfile:
            data_output = pkl.load(recordfile)
        if mpi.size != 1:
            mpi.comm.Barrier()
        mpi.check_time(f"Download output data from save folder {inputdic['save']} END")

    else:
        mpi.check_time(f"Initialisation output data INIT")
        law["coord"] = inputdic["coord"]
        law["kind"] = inputdic["kind"]
        data_output = ResultExactLaw(law, data_origin, int(inputdic["Nmax_scale"]), int(inputdic["Nmax_list"]))
        mpi.check_time(f"Initialisation output data END")

    data_output.check(mpi)

    mpi.check_time("Record output data INIT")
    with open(f"{save_folder}data_output_rk{mpi.rank}.pkl", "wb") as recordfile:
        pkl.dump(data_output, recordfile)
    mpi.check_time("Record output data END")
    if mpi.size != 1:
        mpi.comm.Barrier()

    # ## CALCUL LOI EXACTE

    mpi.check_time("Calculation output data INIT")
    useful_at_scale = ValuesUsefulForCalcAtScale(
        data_origin, mpi
    )  # Insertion des données initiales dans le dictionnaire servant pour le calcul à vecteur fixé
    useful_at_scale.check(mpi)
    if mpi.size != 1:
        mpi.comm.Barrier()

    mpi.check_time("Calculation output data BEG")
    vector_dir0 = [
        0,
        0,
        0,
    ]  # vecteur servant à obtenir le cube translaté en z qui sera distribuée aux processurs si parallélisation
    vector = [0, 0, 0]  # vecteur servant à obtenir les translations du cube dans le plan polaire

    # Test pour vérifier s'il est nécessaire de faire une divergence locale et donc de calculer les points autout du point d'intéret
    div = False
    for k in data_output.datadic.keys():
        if "term_div" in k:
            div = True
            continue

    state = data_output.state

    # Boucle sur les plans
    for ind_dir0 in range(state, data_output.scale.N[0]):
        mpi.check_time(f"Calculation output data state {ind_dir0} INIT")

        # Insertion des données déplacées suivant z dans le dictionnaire servant pour le calcul à vecteur fixé
        vector_dir0[2] = data_output.scale.grid["lz"][ind_dir0]
        useful_at_scale.set_data_dir0(data_origin, vector_dir0, mpi)
        mpi.comm.Barrier()

        i = -1
        # Boucle sur les rayons dans les plans polaires
        for lperp in range(data_output.scale.N[1]):

            # Distribution des calculs 1 vecteur => 1 processeur si parallélisation
            for vect in range(len(data_output.scale.grid["listperp"][lperp])):
                i += 1
                if mpi.group_rank == i % mpi.group_size:
                    # Insertion des données déplacées dans le plan polaire dans le dictionnaire servant pour le calcul à vecteur fixé
                    vector[0] = data_output.scale.grid["listperp"][lperp][vect][0]
                    vector[1] = data_output.scale.grid["listperp"][lperp][vect][1]
                    useful_at_scale.set_data_prim(vector)
                    # Calcul à vecteur fixé
                    indices = [ind_dir0, lperp, vect]
                    data_output.fill_result_at_scale(indices, useful_at_scale)

                    if div == True:
                        for d in range(3):
                            for p in [-1, 1]:
                                vector_div = np.copy(vector)
                                vector_div[d] = vector[d] + p
                                useful_at_scale.set_data_prim(vector_div)
                                if p == -1:
                                    p = 0
                                indices = [ind_dir0, lperp, vect, d, p]
                                data_output.calc_at_scale(indices, useful_at_scale, div=True)
        mpi.comm.Barrier()
        mpi.check_time(f"Calculation output data state {ind_dir0} END")
        data_output.state += 1

        mpi.check_time(f"Record output data state {ind_dir0} INIT")
        with open(f"{save_folder}data_output_rk{mpi.rank}.pkl", "wb") as recordfile:
            pkl.dump(data_output, recordfile)
        mpi.check_time(f"Record output data state {ind_dir0} END")

    mpi.check_time(f"Filtre output data INIT")
    for k in data_output.datadic.keys():
        data_output.datadic[k] = np.where(
            np.abs(data_output.datadic[k]) < 1e-15, 0 * data_output.datadic[k], data_output.datadic[k]
        )
    mpi.check_time(f"Filtre output data END")

    mpi.check_time(f"Reduction output data INIT")
    for k in data_output.datadic.keys():
        if mpi.size != 1:
            total = mpi.comm.reduce(data_output.datadic[k], op=mpi.op, root=0)
        if mpi.rank == 0:
            denom = reduce(lambda x, y: x * y, data_origin.scale.N)
            data_output.datadic[k] = total / denom
    mpi.check_time(f"Reduction output data END")

    mpi.check_time(f"Divergence output data INIT")
    if mpi.rank == 0:
        for k in data_output.datadic.keys():
            if k.startswith("div_"):
                case_vec = data_output.scale.c
                local_dict = {
                    "fx": [
                        data_output.datadic["term_" + k][:, :, :, 0, 0],
                        data_output.datadic["term_" + k][:, :, :, 0, -1],
                    ],
                    "fy": [
                        data_output.datadic["term_" + k][:, :, :, 1, 0],
                        data_output.datadic["term_" + k][:, :, :, 1, -1],
                    ],
                    "fz": [
                        data_output.datadic["term_" + k][:, :, :, 2, 0],
                        data_output.datadic["term_" + k][:, :, :, 2, -1],
                    ],
                }
                local_dict["dx"] = Math_Tools.cdiff(
                    local_dict["fx"], length_case=case_vec[0], precision=2, period=False, point=True
                )
                local_dict["dy"] = Math_Tools.cdiff(
                    local_dict["fy"], length_case=case_vec[1], precision=2, period=False, point=True
                )
                local_dict["dz"] = Math_Tools.cdiff(
                    local_dict["fz"], length_case=case_vec[2], precision=2, period=False, point=True
                )
                data_output.datadic[k] = ne.evaluate(f"dx+dy+dz", local_dict=local_dict)
    mpi.check_time(f"Divergence output data END")

    # for k in data_output.datadic.keys():
    #    if mpi.rank == 0: print(k,' ',np.min(data_output.datadic[k]),' ',np.max(data_output.datadic[k]))
    mpi.check_time("Calculation output data END")

    # ## Enregistrement données finales
    filename = inputdic["folder_record"] + inputdic["name_data"] + "_" + inputdic["name_result"] + ".h5"
    mpi.check_time(f"Record final result in {filename} INIT")
    if mpi.rank == 0:
        data_output.record_to_h5file(filename)
        with h5.File(filename, "a") as file:
            file.create_group("param_origin")
            for k in data_origin.param.keys():
                file["param_origin"].create_dataset(k, data=data_origin.param[k])

    mpi.check_time(f"Record final result in {datafile} END")
    exit()
