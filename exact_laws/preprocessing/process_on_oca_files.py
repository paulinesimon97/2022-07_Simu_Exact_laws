import sys
import numpy as np
import numexpr as ne
import h5py as h5
from datetime import datetime
from contextlib import redirect_stdout
from .. import logging
import configparser

from ..exact_laws_calc.laws import LAWS
from ..exact_laws_calc.terms import TERMS
from .quantities import QUANTITIES
from . import process_on_standard_h5_file


def extract_simu_param_from_OCA_file(file, dic_param, param):
    dic_param["L"] = np.array([file[f"{param}/x"][-1], file[f"{param}/y"][-1], file[f"{param}/z"][-1]])
    dic_param["N"] = np.array(
        [
            np.shape(file[f"{param}/x"])[0],
            np.shape(file[f"{param}/y"])[0],
            np.shape(file[f"{param}/z"])[0],
        ]
    )
    dic_param["c"] = np.array([file[f"{param}/x"][1], file[f"{param}/y"][1], file[f"{param}/z"][1]])
    return dic_param


def extract_quantities_from_OCA_file(file, list_quant, cycle):
    list_data = []
    for quant in list_quant:
        list_data.append(np.transpose(np.ascontiguousarray(file[f"{cycle}/{quant}"], dtype=np.float64)))
    return list_data


def list_quantities(laws, terms, quantities):
    quantities = quantities.copy()
    for term in terms:
        quantities += TERMS[term].variables()
    for law in laws:
        quantities += LAWS[law].variables()
    return list(set(quantities))


def from_OCA_files_to_standard_h5_file(
    input_folder, output_folder, name, cycle, laws, terms, quantities, sim_type, physical_params, reduction
):
    """
    Input: inputdic (Dictionnaire créé par Data_process.inputfile_to_dict())
    Ce qui est fait:
        - Ouverture les fichiers .h5 contenant les données données par l'OCA
        - Calcul des quantités voulues et indiquées dans inputdic
        - Enregistrements des tableaux de données dans un nouveau fichier .h5
        - Print du contenu de ce nouveau fichier via la fonction Data_process.check()
    En cours d'exécution: Print d'informations à propos des étapes intermédiaires.
    Output: nom du nouveau fichier .h5
    """

    output_file = f"{output_folder}/{name}.h5"

    message = (
        f"Begin process from_OCA_files_to_standard_h5_file() with config:"
        f"\n\t - input_folder: {input_folder}"
        f"\n\t - sim_type: {sim_type}"
        f"\n\t - cycle: {input_folder}"
        f"\n\t - output_file: {output_file}"
        f"\n\t - laws: {laws}"
        f"\n\t - terms: {terms}"
        f"\n\t - quantities: {quantities}"
        f"\n\t - reduction: {reduction}"
    )
    logging.getLogger(__name__).info(message)

    output_file = f"{output_folder}/{name}.h5"

    if process_on_standard_h5_file.verif_file_existence(output_file, "Process impossible."):
        logging.getLogger(__name__).info(f"End process from_OCA_files_to_standard_h5_file()\n")
        return output_file

    g = h5.File(output_file, "w")
    dic_param = {}
    needed_quantities = list_quantities(laws, terms, quantities)

    dic_quant = {}
    # param source file (obtained in velocity source file)
    with h5.File(f"{input_folder}/3Dfields_v.h5", "r") as fv:
        if "CGL3" in sim_type:
            dic_param = extract_simu_param_from_OCA_file(fv, dic_param, "3Dgrid")
        else:
            dic_param = extract_simu_param_from_OCA_file(fv, dic_param, "Simulation_Parameters")
    logging.getLogger(__name__).info(f"... End extracting param")

    # velocity source file
    with h5.File(f"{input_folder}/3Dfields_v.h5", "r") as fv:
        (
            dic_quant["vx"],
            dic_quant["vy"],
            dic_quant["vz"],
        ) = extract_quantities_from_OCA_file(fv, ["vx", "vy", "vz"], cycle)
    accessible_quantities = ["v", "w", "gradv", "divv"]
    for aq in accessible_quantities:
        if aq in needed_quantities:
            QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
    del (dic_quant["vx"], dic_quant["vy"], dic_quant["vz"])
    logging.getLogger(__name__).info(f"... End computing quantities from _v.h5")

    # Density source file
    with h5.File(f"{input_folder}/3Dfields_rho.h5", "r") as frho:
        dic_quant["rho"] = extract_quantities_from_OCA_file(
            frho,
            [
                "rho",
            ],
            cycle,
        )
    accessible_quantities = ["rho", "gradrho"]
    for aq in accessible_quantities:
        if aq in needed_quantities:
            QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
    logging.getLogger(__name__).info(f"... End computing quantities from _rho.h5")

    # Pressure source file
    with h5.File(f"{input_folder}/3Dfields_pi.h5", "r") as fp:
        dic_quant["ppar"], dic_quant["pperp"] = extract_quantities_from_OCA_file(fp, ["pparli", "pperpi"], cycle)
    accessible_quantities = ["Ipgyr", "pgyr", "ugyr", "piso", "uiso", "graduiso"]
    for aq in accessible_quantities:
        if aq in needed_quantities:
            QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
    del (dic_quant["ppar"], dic_quant["pperp"])
    logging.getLogger(__name__).info(f"... End computing quantities from _pi.h5")

    # Magnetic field source file
    with h5.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
        (
            dic_quant["bx"],
            dic_quant["by"],
            dic_quant["bz"],
        ) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)
    accessible_quantities = ["Ib", "b", "divb", "Ij", "j", "divj", "Ipm", "pm"]
    for aq in accessible_quantities:
        if aq in needed_quantities:
            QUANTITIES[aq].create_datasets(g, dic_quant, dic_param)
    del dic_quant
    logging.getLogger(__name__).info(f"... End computing quantities from _b.h5")

    # Param
    g.create_group("param")
    for key in dic_param.keys():
        g["param"].create_dataset(key, data=dic_param[key])

    g["param"].create_dataset("laws", data=laws)
    g["param"].create_dataset("quantities", data=quantities)
    g["param"].create_dataset("terms", data=terms)
    g["param"].create_dataset("cycle", data=cycle)
    g["param"].create_dataset("name", data=name)
    g["param"].create_dataset("sim_type", data=sim_type)
    g["param"].create_dataset("reduction", data=reduction)

    for key in physical_params.keys():
        g["param"].create_dataset(key, data=physical_params[key])

    g.close()
    logging.getLogger(__name__).info(f"End process from_OCA_files_to_standard_h5_file()\n")

    return output_file

def reformat_oca_files(config_file):
    """
    config_file example: 
        [INPUT_DATA]
        path = /home/jeandet/Documents/DATA/Pauline/
        cycle = cycle_0
        sim_type = OCA_CGL2

        [OUTPUT_DATA]
        path = ./
        name = OCA_CGL2_cycle0_completeInc
        reduction = 2
        laws = ['SS22I', 'BG17']
        terms = ['flux_dvdvdv']
        quantities = ['Iv']

        [PHYSICAL_PARAMS]
        di = 1
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    
    file_process = from_OCA_files_to_standard_h5_file(
        input_folder=config["INPUT_DATA"]["path"],
        output_folder=config["OUTPUT_DATA"]["path"],
        name=config["OUTPUT_DATA"]["name"],
        sim_type=config["INPUT_DATA"]["sim_type"],
        cycle=config["INPUT_DATA"]["cycle"],
        quantities=eval(config["OUTPUT_DATA"]["quantities"]),
        laws=eval(config["OUTPUT_DATA"]["laws"]),
        terms=eval(config["OUTPUT_DATA"]["terms"]),
        reduction=1,
        physical_params={k: float(eval((config["PHYSICAL_PARAMS"][k]))) for k in config["PHYSICAL_PARAMS"].keys()},
    )
    process_on_standard_h5_file.check_file(file_process)
    
    if config["OUTPUT_DATA"]["reduction"] != "1":
        file_process = process_on_standard_h5_file.data_binning(file_process, int(config["OUTPUT_DATA"]["reduction"]))
        process_on_standard_h5_file.check_file(file_process)
