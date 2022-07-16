import logging
import argparse
import configparser
from exact_laws.running_tools.run_config_wrap import load
from exact_laws.running_tools.backup_wrap import Backup
from exact_laws.exact_laws_calc import calc_exact_laws_from_config

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config-file", help="config file", default='example_input_calc.txt')
parser.add_argument("-q", "--list-laws", help="List available laws", action="store_true")
args = parser.parse_args()

version = "09/07/2022"

if __name__ == "__main__":
    
    if args.list_laws:
        from exact_laws.exact_laws_calc.laws import  LAWS
        print(list(LAWS.keys()))
        exit(0)
    
    config = configparser.ConfigParser()
    config.read(args.config_file)
    try: 
        run_config = load(config['RUN_PARAMS']['config'],bool(eval(config['RUN_PARAMS']['numbap'])))
    except: 
        run_config = load(eval('NOP',False))
    run_config.configure_log('calc_exact_law')
    
    # configure the potential parallelisation process (add old way params)
    if run_config.config == 'OLD':
        run_config.set_nblayer(int(eval(config['RUN_PARAMS']['nblayer'])))
        run_config.set_bufnum(int(eval(config['RUN_PARAMS']['nbbuf'])))

    # configure the saving process (always valid way)
    backup = Backup()
    backup.configure(eval(config['RUN_PARAMS']["save"]), run_config.time_deb, run_config.rank)

    logging.info(f"Run of {__file__} version {version}\n")
    calc_exact_laws_from_config(config_file=args.config_file,run_config=run_config, backup=backup)
    logging.info(f"Exit")
