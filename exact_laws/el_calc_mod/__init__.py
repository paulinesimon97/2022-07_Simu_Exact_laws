import logging
import configparser
import importlib
import os

from .datasets import load_from_standard_file
from .grids import load_incgrid_from_grid
from ..preprocessing import process_on_standard_h5_file


def initialise_original_dataset(input_filename):
    logging.info("INIT Initialisation original data")
    original_dataset, laws, terms = load_from_standard_file(input_filename)
    logging.info("END Initialisation original data ")
    return original_dataset, laws, terms


def calc_exact_laws_from_config(config_file, run_config, backup):
    '''
    config_file example:
        [INPUT_DATA]
        path = ../../data_process/TestEff_CGL2/
        name = OCA_CGL2_cycle0_TestEff_bin2

        [OUTPUT_DATA]
        path = ./
        name = EL_logcyl40_cls100

        [RUN_PARAMS]
        config = NOP
        numbap = O
        nblayer = 8
        nbbuf = 4
        backup = None

        [GRID_PARAMS]
        Nmax_scale = 40
        Nmax_list = 100
        kind = cls
        coord = logcyl
        
        [METHOD_PARAMS]
        method = fourier ou incremental
    '''

    config = configparser.ConfigParser()
    config.read(config_file)

    # translate information useful for the computation
    input_filename = f"{config['INPUT_DATA']['path']}/{config['INPUT_DATA']['name']}.h5"
    output_filename = f"{config['OUTPUT_DATA']['path']}/{config['INPUT_DATA']['name']}_{config['OUTPUT_DATA']['name']}.h5"
    
    method = config['METHOD_PARAMS']['method']
    mod = importlib.import_module(f"exact_laws.el_calc_mod.{method}", "*")
    
    message = (
        f"Begin calc exact law with config:"
        f"\n\t - input_file: {input_filename}"
        f"\n\t - output_file: {output_filename}"
        f"\n\t - method: {method} on {os.cpu_count()} cpu"
    )
    logging.info(message)
    
    # Init Original_dataset
    #if backup.already:
    #    original_dataset = backup.download('data_origin')
    #else:
    #    original_dataset, laws, terms = initialise_original_dataset(input_filename)
    #    if run_config.rank == 0:
    #        backup.save(original_dataset, 'data_origin')
    original_dataset, laws, terms = initialise_original_dataset(input_filename)
    original_dataset.check('original_dataset')
    
    # Init Incremental_grid
    input_grid = {}
    input_grid["coord"] = config['GRID_PARAMS']["coord"]
    input_grid["kind"] = config['GRID_PARAMS']["kind"]
    if method == 'incremental':
        input_grid["Nmax_scale"] = int(eval(config['GRID_PARAMS']["Nmax_scale"]))
        input_grid["Nmax_list"] = int(eval(config['GRID_PARAMS']["Nmax_list"]))

    if backup.already:
        incremental_grid = backup.download('inc_grid')
    else:
        incremental_grid = load_incgrid_from_grid(original_grid=original_dataset.grid, **input_grid)
        if run_config.rank == 0:
            backup.save(incremental_grid, 'inc_grid')
    incremental_grid.check('incremental_grid')

    # ## CALCUL LOI EXACTE
    mod.apply_method(original_dataset, incremental_grid, input_grid["coord"], laws, terms, output_filename, run_config, backup)
    
    if run_config.rank == 0:
        process_on_standard_h5_file.check_file(output_filename)
    
