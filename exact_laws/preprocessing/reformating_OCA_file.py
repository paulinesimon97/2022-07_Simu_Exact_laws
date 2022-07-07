import sys
import numpy as np
import numexpr as ne
import h5py as h5
from datetime import datetime

import math.derivation as Math_Tools
import scan_file


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


def creat_dic_want(inputdic):
    dic_want = {
        "v": False,
        "w": False,
        "gradv": False,
        "delv": False,
        "rho": False,
        "delrho": False,
        "Igyrp": False,
        "gyrp": False,
        "gyru": False,
        "isop": False,
        "isou": False,
        "delisou": False,
        "Ib": False,
        "b": False,
        "delb": False,
        "Ij": False,
        "j": False,
        "delj": False,
        "Ipm": False,
        "pm": False,
    }
    if inputdic["BG17"]:
        dic_want["v"] = True
        dic_want["w"] = True
        dic_want["Ij"] = True
        dic_want["Ib"] = True
    if inputdic["BG17Hall"]:
        dic_want["Ij"] = True
        dic_want["Ib"] = True
    if inputdic["SS22I"]:
        dic_want["v"] = True
        dic_want["Ib"] = True
    if inputdic["SS22IGyr"]:
        dic_want["gradv"] = True
        dic_want["Igyrp"] = True
        dic_want["Ib"] = True
        dic_want["Ipm"] = True
    if inputdic["SS22IHall"]:
        dic_want["Ij"] = True
        dic_want["Ib"] = True
    if inputdic["SS22C"]:
        dic_want["v"] = True
        dic_want["delv"] = True
        dic_want["rho"] = True
        dic_want["delrho"] = True
        dic_want["b"] = True
        dic_want["delb"] = True
        dic_want["pm"] = True
    if inputdic["SS22CIso"]:
        dic_want["v"] = True
        dic_want["delv"] = True
        dic_want["rho"] = True
        dic_want["delrho"] = True
        dic_want["isop"] = True
        dic_want["isou"] = True
    if inputdic["SS22CGyr"]:
        dic_want["v"] = True
        dic_want["gradv"] = True
        dic_want["delv"] = True
        dic_want["rho"] = True
        dic_want["delrho"] = True
        dic_want["gyrp"] = True
        dic_want["gyru"] = True
        dic_want["b"] = True
        dic_want["pm"] = True
    if inputdic["SS22CHall"]:
        dic_want["rho"] = True
        dic_want["j"] = True
        dic_want["delj"] = True
        dic_want["b"] = True
        dic_want["delb"] = True
    if inputdic["SS21C"]:
        dic_want["v"] = True
        dic_want["delv"] = True
        dic_want["rho"] = True
        dic_want["b"] = True
        dic_want["delb"] = True
        dic_want["pm"] = True
    if inputdic["SS21CIso"]:
        dic_want["v"] = True
        dic_want["delv"] = True
        dic_want["rho"] = True
        dic_want["isop"] = True
        dic_want["isou"] = True
        dic_want["delisou"] = True
        dic_want["pm"] = True
    return dic_want


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
    wx, wy, wz = Math_Tools.rot(
        [dic_quant[f"{inc}vx"], dic_quant[f"{inc}vy"], dic_quant[f"{inc}vz"]],
        dic_param["c"],
        precision=4,
        period=True,
    )
    file.create_dataset(f"{inc}wx", data=wx, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}wy", data=wy, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}wz", data=wz, shape=dic_param["N"], dtype=np.float64)


def record_gradv(file, dic_quant, dic_param, inc=""):
    dxvx, dyvx, dzvx = Math_Tools.grad(dic_quant[f"{inc}vx"], dic_param["c"], precision=4, period=True)
    file.create_dataset(f"{inc}dxvx", data=dxvx, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dyvx", data=dyvx, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dzvx", data=dzvx, shape=dic_param["N"], dtype=np.float64)
    dxvy, dyvy, dzvy = Math_Tools.grad(dic_quant[f"{inc}vy"], dic_param["c"], precision=4, period=True)
    file.create_dataset(f"{inc}dxvy", data=dxvy, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dyvy", data=dyvy, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dzvy", data=dzvy, shape=dic_param["N"], dtype=np.float64)
    dxvz, dyvz, dzvz = Math_Tools.grad(dic_quant[f"{inc}vz"], dic_param["c"], precision=4, period=True)
    file.create_dataset(f"{inc}dxvz", data=dxvz, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dyvz", data=dyvz, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset(f"{inc}dzvz", data=dzvz, shape=dic_param["N"], dtype=np.float64)


def record_delv(file, dic_quant, dic_param, inc=""):
    delv = Math_Tools.div(
        [dic_quant[f"{inc}vx"], dic_quant[f"{inc}vy"], dic_quant[f"{inc}vz"]],
        dic_param["c"],
        precision=4,
        period=True,
    )
    file.create_dataset(f"{inc}delv", data=delv, shape=dic_param["N"], dtype=np.float64)


def record_rho(file, dic_quant, dic_param):
    file.create_dataset("rho", data=dic_quant["rho"], shape=dic_param["N"], dtype=np.float64)


def record_delrho(file, dic_quant, dic_param):
    delrhox, delrhoy, delrhoz = Math_Tools.grad(dic_quant["rho"], dic_param["c"], precision=4, period=True)
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
    deluisox, deluisoy, deluisoz = Math_Tools.grad(uiso, dic_param["c"], precision=4, period=True)
    file.create_dataset("duisox", data=deluisox, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset("duisoy", data=deluisoy, shape=dic_param["N"], dtype=np.float64)
    file.create_dataset("duisoz", data=deluisoz, shape=dic_param["N"], dtype=np.float64)


def record_Ij(file, dic_quant, dic_param):
    jx, jy, jz = Math_Tools.rot(
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
    jx, jy, jz = Math_Tools.rot(
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
    jx, jy, jz = Math_Tools.rot(
        [dic_quant["bx"], dic_quant["by"], dic_quant["bz"]],
        dic_param["c"],
        precision=4,
        period=True,
    )
    delj = Math_Tools.div(
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
    delb = Math_Tools.div(tab, dic_param["c"], precision=4, period=True)
    file.create_dataset(f"{inc}delb", data=delb, shape=dic_param["N"], dtype=np.float64)


def record_pm(file, dic_quant, dic_param):
    file.create_dataset(
        "pm",
        data=ne.evaluate("(bx*bx+by*by+bz*bz)/2/rho", local_dict=dic_quant),
        shape=dic_param["N"],
        dtype=np.float64,
    )


def data_process_OCA(inputdic):
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

    file_record = f"{inputdic['folder_record']}{inputdic['name_record']}.h5"
    if scan_file.verif_file_existence(file_record, "Data process impossible."):
        return file_record

    g = h5.File(file_record, "w")
    dic_param = {}
    dic_want = creat_dic_want(inputdic)

    dic_quant = {}
    # kinetic source file
    with h5.File(f"{inputdic['folder_data']}3Dfields_v.h5", "r") as fv:
        if "CGL_3" in inputdic["folder_data"]:
            dic_param = extract_simu_param_from_OCA_file(fv, dic_param, "3Dgrid")
        else:
            dic_param = extract_simu_param_from_OCA_file(fv, dic_param, "Simulation_Parameters")
        (
            dic_quant["vx"],
            dic_quant["vy"],
            dic_quant["vz"],
        ) = extract_quantities_from_OCA_file(fv, ["vx", "vy", "vz"], inputdic["cycle"])
    if dic_want["v"]:
        record_v(g, dic_quant, dic_param)
    if dic_want["w"]:
        record_w(g, dic_quant, dic_param)
    if dic_want["gradv"]:
        record_gradv(g, dic_quant, dic_param)
    if dic_want["delv"]:
        record_delv(g, dic_quant, dic_param)
    del (dic_quant["vx"], dic_quant["vy"], dic_quant["vz"])
    print(f"\t - End data process v: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')}")
    sys.stdout.flush()

    # Density source file
    with h5.File(f"{inputdic['folder_data']}3Dfields_rho.h5", "r") as frho:
        dic_quant["rho"] = extract_quantities_from_OCA_file(
            frho,
            [
                "rho",
            ],
            inputdic["cycle"],
        )
    if dic_want["rho"]:
        record_rho(g, dic_quant, dic_param)
    if dic_want["delrho"]:
        record_delrho(g, dic_quant, dic_param)
    print(f"\t - End data process rho: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')}")
    sys.stdout.flush()

    # Pressure source file
    with h5.File(f"{inputdic['folder_data']}3Dfields_pi.h5", "r") as fp:
        dic_quant["ppar"], dic_quant["pperp"] = extract_quantities_from_OCA_file(
            fp, ["pparli", "pperpi"], inputdic["cycle"]
        )
    if dic_want["Igyrp"]:
        record_Igyrp(g, dic_quant, dic_param)
    if dic_want["gyrp"]:
        record_gyrp(g, dic_quant, dic_param)
    if dic_want["gyru"]:
        record_gyru(g, dic_quant, dic_param)
    if dic_want["isop"]:
        record_isop(g, dic_quant, dic_param)
    if dic_want["isou"]:
        record_isou(g, dic_quant, dic_param)
    if dic_want["delisou"]:
        record_delisou(g, dic_quant, dic_param)
    del (dic_quant["ppar"], dic_quant["pperp"])
    print(f"\t - End data process p: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')}")
    sys.stdout.flush()

    # Magnetic field source file
    with h5.File(f"{inputdic['folder_data']}3Dfields_b.h5", "r") as fb:
        (
            dic_quant["bx"],
            dic_quant["by"],
            dic_quant["bz"],
        ) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], inputdic["cycle"])
    if dic_want["Ib"]:
        record_Ib(g, dic_quant, dic_param)
    if dic_want["b"]:
        record_b(g, dic_quant, dic_param)
    if dic_want["delb"]:
        record_delb(g, dic_quant, dic_param)
    if dic_want["Ij"]:
        record_Ij(g, dic_quant, dic_param)
    if dic_want["j"]:
        record_j(g, dic_quant, dic_param)
    if dic_want["delj"]:
        record_delj(g, dic_quant, dic_param)
    if dic_want["Ipm"]:
        record_Ipm(g, dic_quant, dic_param)
    if dic_want["pm"]:
        record_pm(g, dic_quant, dic_param)
    del dic_quant
    print(f"\t - End data process b: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')}")
    sys.stdout.flush()

    # Param
    g.create_group("param")
    for key in dic_param.keys():
        g["param"].create_dataset(key, data=dic_param[key])
    for key in inputdic.keys():
        if not ("folder" in key or "name" in key):
            g["param"].create_dataset(key, data=inputdic[key])
    g.close()
    print(f"Data process end: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')} \n")
    scan_file.check_h5_file_with_good_format(file_record)
    print(" ")
    return file_record


def inputfile_to_dict(filename):
    """
    Input: filename(str, name+ext of the input .txt file)
    Read the input txt file
    Formating of the information
    Display the information
    Output: dic that contains the input information such as usable by the Data_process functions
    """
    inputdic = {}
    with open(filename, encoding="utf-8") as entree:
        for line in entree:
            value = line.split()
            if len(value) >= 2:
                inputdic[value[0]] = value[1]
    inputdic["folder_data"] = str(inputdic["folder_data"])
    inputdic["cycle"] = str(inputdic["cycle"])
    inputdic["folder_record"] = str(inputdic["folder_record"])
    inputdic["name_record"] = str(inputdic["name_record"])
    inputdic["BG17"] = bool(int(inputdic["BG17"]))
    inputdic["BG17Hall"] = bool(int(inputdic["BG17Hall"]))
    inputdic["SS22I"] = bool(int(inputdic["SS22I"]))
    inputdic["SS22IGyr"] = bool(int(inputdic["SS22IGyr"]))
    inputdic["SS22IHall"] = bool(int(inputdic["SS22IHall"]))
    inputdic["SS22C"] = bool(int(inputdic["SS22C"]))
    inputdic["SS22CIso"] = bool(int(inputdic["SS22CIso"]))
    inputdic["SS22CGyr"] = bool(int(inputdic["SS22CGyr"]))
    inputdic["SS22CHall"] = bool(int(inputdic["SS22CHall"]))
    inputdic["SS21C"] = bool(int(inputdic["SS21C"]))
    inputdic["SS21CIso"] = bool(int(inputdic["SS21CIso"]))
    inputdic["SS21CHom"] = bool(int(inputdic["SS21CHom"]))
    inputdic["di"] = float(inputdic["di"])
    inputdic["bin"] = int(inputdic["bin"])
    print(f"input_process.txt content:")
    for k in inputdic.keys():
        print(f"\t - {k}: {inputdic[k]}")
    print()
    sys.stdout.flush()
    return inputdic
