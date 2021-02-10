import pickle
import platform
print(platform.architecture())
#-----------------------------------------IO-----------------------------

import platform
platform.architecture()


def load_any_obj_pkl(path):
    ''' load any object from pickle file
    '''
    with open(path, 'rb') as f:
        any_obj = pickle.load(f)
    return any_obj

def save_any_obj_pkl(obj, path):
    ''' save any object to pickle file
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def write_csv(all_infos, name, delimiter = ','):
    import csv
    with open(name, "w",newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(all_infos)

def read_csv(file_name, delimiter = '\t'):
    import csv
    all_infos = []
    with open(file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        for row in spamreader:
            all_infos.append(row)
    return all_infos