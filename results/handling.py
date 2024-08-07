import os


def get_pathname(ext, *kwargs):
    results_filename = ""

    for arg in kwargs:
        # if args contains spaces, replace them with nothing
        if " " in arg:
            arg = arg.replace(" ", "")
            # if args contains ( brackets, maintain only the content inside the brackets
        if "(" in arg:
            arg = arg[arg.index("(") + 1:arg.index(")")]
        results_filename = results_filename + arg + "__"
    results_filename = results_filename[:-2] + "." + ext

    return results_filename


def get_info_from_filename(filename):
    features = 'Unknown'
    filename_splits = os.path.basename(filename).split('__')

    img_type = filename_splits[1].split('_')[2]
    scaler_name = filename_splits[2]
    features = filename_splits[-2]
    alg_name = filename_splits[-1][:-4]

    return img_type, scaler_name, features, alg_name


