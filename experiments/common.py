import os.path

import pandas as pd

from experiments.anomaly import perform_anomaly
from experiments.classification import perform_classification
from experiments.regression import perform_regression
from preprocessing.preprocess import exec_cleaning, keep_data_until_target, extend_target
from results.handling import get_pathname
from results.common import create_empty_df_from_task, add_results_to_df_by_task
import results.regression as reg
import results.classification as cla
from utils.files import get_csv_files, count_files_with_pattern
from utils.folders import make_path, file_exists
from utils.print import print_info
from utils.saves import save_info


def run_experiment(folder_handler, set_folder, img_types_path, scalers, algorithms, task, *kwargs):

    n_experiments = len(img_types_path) * len(scalers) * len(algorithms) * len(get_csv_files(img_types_path[0]))
    #n_experiments = len(scalers) * len(algorithms) * len(get_csv_files(img_types_path[0]))

    results_pickle_path = make_path(folder_handler.results_pickle, set_folder + '.pickle')
    if count_files_with_pattern(folder_handler.results, set_folder) == n_experiments and file_exists(results_pickle_path):
        print("Results already computed. Loading: " + results_pickle_path)
        results = pd.read_pickle(results_pickle_path)
    else:
        print("Some results are missing. Now computing")
        results = handle_experiment(n_experiments, folder_handler, set_folder, img_types_path, scalers, algorithms,
                                    task, *kwargs)
    return results


def handle_experiment(n_experiments, folder_handler, set_folder, img_types_path, scalers, algorithms, task, *kwargs):

    i = 0
    results = create_empty_df_from_task(task)
    results_std = create_empty_df_from_task(task)

    for img_type_path in img_types_path:
        name_img_type = os.path.basename(img_type_path)
        for scaler in scalers:
            name_scaler = scaler.__class__.__name__

            features = get_csv_files(img_type_path)
            for feature in features:
                name_features = os.path.splitext(os.path.basename(feature))[0]
                feature_path = make_path(img_type_path, feature)
                data = pd.read_csv(feature_path)

                data = exec_cleaning(data)   # Data cleaning
                #data = keep_data_until_target(data)  # Keep only timestamps until target
                #data = extend_target(data)  # Extend target to include up to 15 seconds before or after target

                # Perform classification/regression for each algorithm
                for alg_name, alg in algorithms.items():
                    print_info(name_img_type, name_scaler, alg_name, name_features)

                    results_filename = get_pathname('csv', set_folder, name_img_type, name_scaler, name_features, alg_name)
                    plot_filename = get_pathname('png', set_folder, name_img_type, name_scaler, name_features, alg_name)
                    results_path = make_path(folder_handler.results, results_filename)
                    results_std_path = make_path(folder_handler.results_std, results_filename)
                    plot_path = make_path(folder_handler.plots, plot_filename)

                    if file_exists(results_path):
                        print('Results already exist for', results_filename)
                        results = add_results_to_df_by_task(task, results, results_path)

                    else:
                        save_info(task, img_type_path, scaler, feature, alg_name, alg)

                        if task == 'Regression':
                            regression_type = kwargs[0]
                            curr_results, plot = perform_regression(data, scaler, alg, regression_type, print_results=False)
                            reg.save_results_to_disk(curr_results, results_path)
                            reg.save_plot_to_disk(plot, plot_path)

                        elif task == 'Classification':
                            cross_val = kwargs[0]
                            loo = kwargs[1]
                            class_imbalance_handling = kwargs[2]
                            curr_results, curr_std = perform_classification(data, scaler, alg, cross_val, loo, class_imbalance_handling)
                            cla.save_results_to_disk(curr_results, results_path)
                            cla.save_results_to_disk(curr_std, results_std_path)
                            #acc_score, class_report, class_report_dict = perform_classification(data, scaler, alg, cross_val, class_imbalance_handling)

                        elif task in ['Anomaly', 'AnomalyClassic']:
                            cross_val = kwargs[0]
                            loo = kwargs[1]
                            curr_results, curr_std = perform_anomaly(data, scaler, alg, cross_val, loo, task)
                            cla.save_results_to_disk(curr_results, results_path)
                            cla.save_results_to_disk(curr_std, results_std_path)

                    results = add_results_to_df_by_task(task, results, results_path)
                    if task != 'Regression':
                        results_std = add_results_to_df_by_task(task, results_std, results_std_path)
                    i += 1
                    if i == n_experiments:
                        results_pickle_path = make_path(folder_handler.results_pickle, set_folder + '.pickle')
                        results_std_pickle_path = make_path(folder_handler.results_pickle, set_folder + '_std.pickle')

                        results.to_pickle(results_pickle_path)
                        if task != 'Regression':
                            results_std.to_pickle(results_std_pickle_path)

    return results


