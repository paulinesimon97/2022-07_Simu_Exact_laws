import h5py as h5
import os


def recursive_copy_of_file(file_to_copy, output_file, path='/', input_to_record={}):
    for k in file_to_copy[path].keys():
        splitted_key = str(file_to_copy[path + '/' + k]).split()
        if splitted_key[1] == 'group':
            output_file[path].create_group(k)
            recursive_copy_of_file(file_to_copy, output_file, path + '/' + k)
        else:
            output_file[path].create_dataset(k, data=input_to_record.get(k, h5.Empty('f')))


def copy_struct_h5file(name_file_to_copy, name_output_file, input_to_record={}):
    with h5.File(name_file_to_copy, 'r') as fc:
        with h5.File(name_output_file, 'w') as fo:
            recursive_copy_of_file(fc, fo, input_to_record=input_to_record)


def copy_struct_folder_of_h5file(name_folder, name_output_folder, input_to_record={}):
    os.mkdir(name_output_folder)
    for name_file in os.listdir(name_folder):
        if name_file.endswith('.h5'):
            copy_struct_h5file(name_folder + name_file, name_output_folder + name_file, input_to_record=input_to_record)
