import os
import glob
import pandas as pd

from results.handling import get_info_from_filename


##
# @brief This function returns a list of all csv files in a directory
# @param path: path of the directory
# @return list of csv files in the directory
def get_csv_files(path):
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    return files


##
# @brief This function saves a python dictionary to a csv file
# @param results: dictionary to save
# @param path: path of the csv file
def save_dict_to_csv(results, path):
    df = pd.DataFrame.from_dict(results)
    df.to_csv(path, index=False, header=True)


def count_files_with_pattern(path, pattern):
    count = 0
    for filename in glob.glob(os.path.join(path, '*')):
        if pattern in filename:
            count += 1
    return count

