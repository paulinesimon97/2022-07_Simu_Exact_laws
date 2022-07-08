import sys
import numpy as np
import numexpr as ne
import h5py as h5
from datetime import datetime
from contextlib import redirect_stdout
import logging
import configparser

from ..math import derivation
from ..exact_laws_calc.laws import LAWS
from .quantities import QUANTITIES
from . import scan_file

version = "27/06/2022"


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


def record_v(file, dic_quant, dic_param, inc=""):
    file.create_dataset(
        f"{inc}vx",
        data=dic_quant[f"{inc}vx"],
        shape=dic_param["N"],
        dtype=np.float64,
    )
    file.create_dataset(
        f"{inc}vy",
        data=dic_quant[f"{inc}vy"],
        shape=dic_param["N"],
        dtype=np.float64,
    )
    file.create_dataset(
        f"{inc}vz",
        data=dic_quant[f"{inc}vz"],
        shape=dic_param["N"],
        dtype=np.float64,
    )


def record_w(file, dic_quant, dic_param, inc=""):
    wx, wy, wz = derivation.rot(
        [dic_quant[f"{inc}vx"], dic_quant[f"{inc}vy"], dic_quant[f"{inc}vz"]],
        dic_param["c"],
        precision=4,
        period=True,
    )
    file.create_dataset(f"{inc}wx", data=wx, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}wy", data=wy, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}wz", data=wz, shape=dic_param["N"], dtype=np.float64)


def record_gradv(file, dic_quant, dic_param, inc=""):
    dxvx, dyvx, dzvx = derivation.grad(dic_quant[f"{inc}vx"], dic_param["c"], precision=4, period=True)
    file.create_dataset(f"{inc}dxvx", data=dxvx, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dyvx", data=dyvx, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dzvx", data=dzvx, shape=dic_param["N"], dtype=np.float64)
    dxvy, dyvy, dzvy = derivation.grad(dic_quant[f"{inc}vy"], dic_param["c"], precision=4, period=True)
    file.create_dataset(f"{inc}dxvy", data=dxvy, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dyvy", data=dyvy, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dzvy", data=dzvy, shape=dic_param["N"], dtype=np.float64)
    dxvz, dyvz, dzvz = derivation.grad(dic_quant[f"{inc}vz"], dic_param["c"], precision=4, period=True)
    file.create_dataset(f"{inc}dxvz", data=dxvz, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dyvz", data=dyvz, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dzvz", data=dzvz, shape=dic_param["N"], dtype=np.float64)


def record_delv(file, dic_quant, dic_param, inc=""):
    delv = derivation.div(
        [dic_quant[f"{inc}vx"], dic_quant[f"{inc}vy"], dic_quant[f"{inc}vz"]],
        dic_param["c"],
        precision=4,
        period=True,
    )
    file.create_dataset(f"{inc}delv", data=delv, shape=dic_param["N"], dtype=np.float64)


def record_rho(file, dic_quant, dic_param):
    file.create_dataset("rho", data=dic_quant["rho"], shape=dic_param["N"], dtype=np.float64)


def record_delrho(file, dic_quant, dic_param):
    delrhox, delrhoy, delrhoz = derivation.grad(dic_quant["rho"], dic_param["c"], precision=4, period=True)
    file.create_dataset("drhox", data=delrhox, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset("drhoy", data=delrhoy, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset("drhoz", data=delrhoz, shape=dic_param["N"], dtype=np.float64)


def record_Igyrp(file, dic_quant, dic_param):
    file.create_dataset("Ippar", data=dic_quant["ppar"], shape=dic_param["N"], dtype=np.float64)
    file.create_dataset("Ipperp", data=dic_quant["pperp"], shape=dic_param["N"], dtype=np.float64)


def record_gyrp(file, dic_quant, dic_param):
    file.create_dataset(
        "ppar",
        data=ne.evaluate("ppar/rho", local_dict=dic_quant),
        shape=dic_param["N"],
        dtype=np.float64,
    )
    file.create_dataset(
        "pperp",
        data=ne.evaluate("pperp/rho", local_dict=dic_quant),
        shape=dic_param["N"],
        dtype=np.float64,
    )


def record_gyru(file, dic_quant, dic_param):
    file.create_dataset(
        "ugyr",
        data=ne.evaluate("(ppar+pperp+pperp)/2/rho", local_dict=dic_quant),
        shape=dic_param["N"],
        dtype=np.float64,
    )


def record_isop(file, dic_quant, dic_param):
    file.create_dataset(
        "piso",
        data=ne.evaluate("(ppar+pperp+pperp)/3/rho", local_dict=dic_quant),
        shape=dic_param["N"],
        dtype=np.float64,
    )


def record_isou(file, dic_quant, dic_param):
    file.create_dataset(
        "uiso",
        data=ne.evaluate("(ppar+pperp+pperp)/2/rho", local_dict=dic_quant),
        shape=dic_param["N"],
        dtype=np.float64,
    )


def record_delisou(file, dic_quant, dic_param):
    uiso = ne.evaluate("(ppar+pperp+pperp)/2/rho", local_dict=dic_quant)
    deluisox, deluisoy, deluisoz = derivation.grad(uiso, dic_param["c"], precision=4, period=True)
    file.create_dataset("duisox", data=deluisox, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset("duisoy", data=deluisoy, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset("duisoz", data=deluisoz, shape=dic_param["N"], dtype=np.float64)


def record_Ij(file, dic_quant, dic_param):
    jx, jy, jz = derivation.rot(
        [dic_quant["bx"], dic_quant["by"], dic_quant["bz"]],
        dic_param["c"],
        precision=4,
        period=True,
    )
    file.create_dataset("Ijx", data=jx, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset("Ijy", data=jy, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset("Ijz", data=jz, shape=dic_param["N"], dtype=np.float64)


def record_j(file, dic_quant, dic_param):
    rho = dic_quant["rho"]
    jx, jy, jz = derivation.rot(
        [dic_quant["bx"], dic_quant["by"], dic_quant["bz"]],
        dic_param["c"],
        precision=4,
        period=True,
    )
    file.create_dataset("jx", data=ne.evaluate("jx/rho"), shape=dic_param["N"], dtype=np.float64)
    file.create_dataset("jy", data=ne.evaluate("jy/rho"), shape=dic_param["N"], dtype=np.float64)
    file.create_dataset("jz", data=ne.evaluate("jz/rho"), shape=dic_param["N"], dtype=np.float64)


def record_delj(file, dic_quant, dic_param):
    rho = dic_quant["rho"]
    jx, jy, jz = derivation.rot(
        [dic_quant["bx"], dic_quant["by"], dic_quant["bz"]],
        dic_param["c"],
        precision=4,
        period=True,
    )
    delj = derivation.div(
        [ne.evaluate("jx/rho"), ne.evaluate("jy/rho"), ne.evaluate("jz/rho")],
        dic_param["c"],
        precision=4,
        period=True,
    )
    file.create_dataset("delj", data=delj, shape=dic_param["N"], dtype=np.float64)


def record_Ib(file, dic_quant, dic_param, inc="I"):
    file.create_dataset(f"{inc}bx", data=dic_quant[f"bx"], shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}by", data=dic_quant[f"by"], shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}bz", data=dic_quant[f"bz"], shape=dic_param["N"], dtype=np.float64)


def record_Ipm(file, dic_quant, dic_param):
    file.create_dataset(
        "Ipm",
        data=ne.evaluate("bx*bx/2+by*by/2+bz*bz/2", local_dict=dic_quant),
        shape=dic_param["N"],
        dtype=np.float64,
    )


def record_b(file, dic_quant, dic_param):
    file.create_dataset(
        "bx",
        data=ne.evaluate("bx/sqrt(rho)", local_dict=dic_quant),
        shape=dic_param["N"],
        dtype=np.float64,
    )
    file.create_dataset(
        "by",
        data=ne.evaluate("by/sqrt(rho)", local_dict=dic_quant),
        shape=dic_param["N"],
        dtype=np.float64,
    )
    file.create_dataset(
        "bz",
        data=ne.evaluate("bz/sqrt(rho)", local_dict=dic_quant),
        shape=dic_param["N"],
        dtype=np.float64,
    )


def record_delb(file, dic_quant, dic_param, inc=""):
    if inc == "I":
        tab = [dic_quant["bx"], dic_quant["by"], dic_quant["bz"]]
    else:
        tab = [
            ne.evaluate("bx/sqrt(rho)", local_dict=dic_quant),
            ne.evaluate("by/sqrt(rho)", local_dict=dic_quant),
            ne.evaluate("bz/sqrt(rho)", local_dict=dic_quant),
        ]
    delb = derivation.div(tab, dic_param["c"], precision=4, period=True)
    file.create_dataset(f"{inc}delb", data=delb, shape=dic_param["N"], dtype=np.float64)


def record_pm(file, dic_quant, dic_param):
    file.create_dataset(
        "pm",
        data=ne.evaluate("(bx*bx+by*by+bz*bz)/2/rho", local_dict=dic_quant),
        shape=dic_param["N"],
        dtype=np.float64,
    )


def list_quantities(laws, quantities):
    quantities = quantities.copy()
    for law in laws:
        quantities += LAWS[law].variables()
    return list(set(quantities))


def data_process_OCA(input_folder, output_folder, name, cycle, laws, quantities, sim_type, di, reduction):
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
    print(f"Data process beginning: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')}")
    sys.stdout.flush()

    file_record = f"{output_folder}/{name}.h5"

    if scan_file.verif_file_existence(file_record, "Data process impossible."):
        return file_record

    g = h5.File(file_record, "w")
    dic_param = {}
    needed_quantities = list_quantities(laws, quantities)

    dic_quant = {}
    # kinetic source file
    with h5.File(f"{input_folder}/3Dfields_v.h5", "r") as fv:
        if "CGL3" in sim_type:
            dic_param = extract_simu_param_from_OCA_file(fv, dic_param, "3Dgrid")
        else:
            dic_param = extract_simu_param_from_OCA_file(fv, dic_param, "Simulation_Parameters")
        (
            dic_quant["vx"],
            dic_quant["vy"],
            dic_quant["vz"],
        ) = extract_quantities_from_OCA_file(fv, ["vx", "vy", "vz"], cycle)
    if "v" in needed_quantities:
        QUANTITIES['v'].create_datasets(g, dic_quant, dic_param)
    if "w" in needed_quantities:
        record_w(g, dic_quant, dic_param)
    if "gradv" in needed_quantities:
        record_gradv(g, dic_quant, dic_param)
    if "delv" in needed_quantities:
        record_delv(g, dic_quant, dic_param)
    del (dic_quant["vx"], dic_quant["vy"], dic_quant["vz"])
    print(f"\t - End data process v: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')}")
    sys.stdout.flush()

    # Density source file
    with h5.File(f"{input_folder}/3Dfields_rho.h5", "r") as frho:
        dic_quant["rho"] = extract_quantities_from_OCA_file(
            frho,
            [
                "rho",
            ],
            cycle,
        )
    if "rho" in needed_quantities:
        record_rho(g, dic_quant, dic_param)
    if "delrho" in needed_quantities:
        record_delrho(g, dic_quant, dic_param)
    print(f"\t - End data process rho: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')}")
    sys.stdout.flush()

    # Pressure source file
    with h5.File(f"{input_folder}/3Dfields_pi.h5", "r") as fp:
        dic_quant["ppar"], dic_quant["pperp"] = extract_quantities_from_OCA_file(
            fp, ["pparli", "pperpi"], cycle
        )
    if "Igyrp" in needed_quantities:
        record_Igyrp(g, dic_quant, dic_param)
    if "gyrp" in needed_quantities:
        record_gyrp(g, dic_quant, dic_param)
    if "gyru" in needed_quantities:
        record_gyru(g, dic_quant, dic_param)
    if "isop" in needed_quantities:
        record_isop(g, dic_quant, dic_param)
    if "isou" in needed_quantities:
        record_isou(g, dic_quant, dic_param)
    if "delisou" in needed_quantities:
        record_delisou(g, dic_quant, dic_param)
    del (dic_quant["ppar"], dic_quant["pperp"])
    print(f"\t - End data process p: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')}")
    sys.stdout.flush()

    # Magnetic field source file
    with h5.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
        (
            dic_quant["bx"],
            dic_quant["by"],
            dic_quant["bz"],
        ) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)
    if "Ib" in needed_quantities:
        QUANTITIES['Ib'].create_datasets(g, dic_quant, dic_param)
    if "b" in needed_quantities:
        QUANTITIES['b'].create_datasets(g, dic_quant, dic_param)
    if "delb" in needed_quantities:
        record_delb(g, dic_quant, dic_param)
    if "Ij" in needed_quantities:
        record_Ij(g, dic_quant, dic_param)
    if "j" in needed_quantities:
        record_j(g, dic_quant, dic_param)
    if "delj" in needed_quantities:
        record_delj(g, dic_quant, dic_param)
    if "Ipm" in needed_quantities:
        record_Ipm(g, dic_quant, dic_param)
    if "pm" in needed_quantities:
        record_pm(g, dic_quant, dic_param)
    del dic_quant
    print(f"\t - End data process b: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')}")
    sys.stdout.flush()

    # Param
    g.create_group("param")
    for key in dic_param.keys():
        g["param"].create_dataset(key, data=dic_param[key])

    g["param"].create_dataset('laws', data=laws)
    g["param"].create_dataset('quantities', data=quantities)
    g["param"].create_dataset('cycle', data=cycle)
    g["param"].create_dataset('name', data=name)
    g["param"].create_dataset('sim_type', data=sim_type)
    g["param"].create_dataset('di', data=di)
    g["param"].create_dataset('reduction', data=reduction)

    g.close()
    print(f"Data process end: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')} \n")
    scan_file.check_h5_file_with_good_format(file_record)
    print(" ")
    return file_record


# def inputfile_to_dict(filename):
#     """
#     Input: filename(str, name+ext of the input .txt file)
#     Read the input txt file
#     Formating of the information
#     Display the information
#     Output: dic that contains the input information such as usable by the Data_process functions
#     """
#     inputdic = {}
#     with open(filename, encoding="utf-8") as entree:
#         for line in entree:
#             value = line.split()
#             if len(value) >= 2:
#                 inputdic[value[0]] = value[1]
#
#     def safe_get_bool(in_dict, key):
#         return bool(int(in_dict.get(key, False)))
#
#     inputdic["folder_data"] = str(inputdic["folder_data"])
#     inputdic["cycle"] = str(inputdic["cycle"])
#     inputdic["folder_record"] = str(inputdic["folder_record"])
#     inputdic["name_record"] = str(inputdic["name_record"])
#     inputdic["BG17"] = safe_get_bool(inputdic, "BG17")
#     inputdic["BG17Hall"] = bool(int(inputdic["BG17Hall"]))
#     inputdic["SS22I"] = bool(int(inputdic["SS22I"]))
#     inputdic["SS22IGyr"] = bool(int(inputdic["SS22IGyr"]))
#     inputdic["SS22IHall"] = bool(int(inputdic["SS22IHall"]))
#     inputdic["SS22C"] = bool(int(inputdic["SS22C"]))
#     inputdic["SS22CIso"] = bool(int(inputdic["SS22CIso"]))
#     inputdic["SS22CGyr"] = bool(int(inputdic["SS22CGyr"]))
#     inputdic["SS22CHall"] = bool(int(inputdic["SS22CHall"]))
#     inputdic["SS21C"] = bool(int(inputdic["SS21C"]))
#     inputdic["SS21CIso"] = bool(int(inputdic["SS21CIso"]))
#     inputdic["SS21CHom"] = bool(int(inputdic["SS21CHom"]))
#     inputdic["di"] = float(inputdic["di"])
#     inputdic["bin"] = int(inputdic["bin"])
#     print(f"input_process.txt content:")
#     for k in inputdic.keys():
#         print(f"\t - {k}: {inputdic[k]}")
#     print()
#     sys.stdout.flush()
#     return inputdic


'''

[INPUT_DATA]
path = /home/jeandet/Documents/DATA/Pauline/
cycle = cycle_0
sim_type = OCA_CGL2

[OUTPUT_DATA]
path = ./
name = OCA_CGL2_cycle0_completeInc
reduction = 2
laws = ['SS22I', 'BG17']
quantities = ['Iv']

[PHYSICAL_PARAMS]
di = 1

'''


def main(config_file):
    now = datetime.today()
    with open(f"output_process_{now.strftime('%d%m%Y_%H%M')}.txt", "w") as f:
        with redirect_stdout(f):
            logging.info(f'Run of {__file__} version {version}')
            sys.stdout.flush()
            config = configparser.ConfigParser()
            config.read(config_file)
            file_process = data_process_OCA(input_folder=config['INPUT_DATA']['path'],
                                            output_folder=config['OUTPUT_DATA']['path'],
                                            name=config['OUTPUT_DATA']['name'],
                                            sim_type=config['INPUT_DATA']['sim_type'],
                                            cycle=config['INPUT_DATA']['cycle'],
                                            quantities=eval(config['OUTPUT_DATA']['quantities']),
                                            laws=eval(config['OUTPUT_DATA']['laws']),
                                            reduction=1,
                                            di=float(eval((config['PHYSICAL_PARAMS']['di'])))
                                            )
            if config['OUTPUT_DATA']["reduction"] != '1':
                scan_file.data_binning(file_process, int(config['OUTPUT_DATA']["reduction"]))
            now = datetime.now()
            logging.info(f"End the {datetime.today().strftime('%d/%m/%Y')} at {datetime.today().strftime('%H:%M')}")
