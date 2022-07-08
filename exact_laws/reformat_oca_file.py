from exact_laws.preprocessing.reformat_OCA_file import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config-file", help="config file", default='input_process.txt')
parser.add_argument("-q", "--list-quantities", help="List available quantities", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    if args.list_quantities:
        from exact_laws.preprocessing.quantities import  QUANTITIES
        print(list(QUANTITIES.keys()))
        exit(0)
    main(config_file=args.config_file)
