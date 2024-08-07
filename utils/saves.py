import os


def save_info(task, folder, scaler_name, feature, alg_name, alg):
    # store the variables in a csv file
    # to be able to resume the experiment
    # if it is interrupted

    with open(os.path.join('saves', task + '.csv'), 'w') as f:
        f.write(f"Folder: {folder}\n")
        f.write(f"Scaler: {scaler_name}\n")
        f.write(f"Feature: {feature}\n")
        f.write(f"AlgorithmName: {alg_name}\n")
        f.write(f"Algorithm: {alg}\n")


def load_info(task):
    # load the variables from the csv file
    # to be able to resume the experiment
    # if it is interrupted

    with open(os.path.join('saves', task + '.csv'), 'r') as f:
        lines = f.readlines()
        folder = lines[0].split(': ')[1].strip()
        scaler_name = lines[1].split(': ')[1].strip()
        feature = lines[2].split(': ')[1].strip()
        alg_name = lines[3].split(': ')[1].strip()
        alg = lines[4].split(': ')[1].strip()

    return folder, scaler_name, feature, alg_name, alg
