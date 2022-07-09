from Exact_laws.exact_laws.preprocessing.process_on_oca_files import reformat_oca_files
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config-file", help="config file", default='example_input_process.ini')
parser.add_argument("-q", "--list-quantities", help="List available quantities", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    if args.list_quantities:
        from exact_laws.preprocessing.quantities import  QUANTITIES
        print(list(QUANTITIES.keys()))
        exit(0)
    reformat_oca_files(config_file=args.config_file)
