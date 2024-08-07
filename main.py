import os

import pandas as pd

from experiments.common import run_experiment
from objects import folder_handler as fh
from results.common import create_empty_df_from_task
from results.latex import make_latex_table_from_df, save_results_to_latex, save_best_n_results_to_latex, \
    save_best_n_results_to_latex_by_filters
from setup.experimental_settings import get_scalers, get_regressors, get_classifiers, get_anomaly_detectors
from utils.folders import get_subfolders, make_path

# Root path
""" Features with a custom, progressive, timestamp (1 - N).
features_root = "../FeaturesEngineered/"
regression_imgs_upto_target = 'regression_imgs_upto_target'     # Tutti i set insieme, immagini fino al TARGET compreso
regression_imgs_all = 'regression_imgs_all'                     # Tutti i set insieme, tutte le immagini
"""

# Settings
EXPS_NAME = ["Regression", "Classification", "Anomaly",
             "AnomalyClassic"]  # NB: "AnomalyClassic" is for reporting NP, NR, NF, PP, PR, PF as metrics
EXP = EXPS_NAME[2]
EXP_FOLDER = 'AnomalyDetection'
REGRESSION_TYPE = "Days"  # "Days" or "Timestamp"

""" Features with the real timestamp."""
features_root = "../../Features/"  # Cheesev9

n = 100000
metric_reg = 'RMSE'
metric_cla = 'bacc'
if EXP == "AnomalyClassic":
    metric_ano = 'PF'
else:
    metric_ano = metric_cla

filters = ["ImT", "Scaler", "Classifier", "Features"]

if __name__ == '__main__':

    scalers = get_scalers()
    set_folders = get_subfolders(features_root)

    # if there are no set folders, then the features are in the root folder
    if not set_folders:
        set_folders = [features_root]

    all_results = create_empty_df_from_task(EXP, "All")
    folder_handler_all = fh.FolderHandler(features_root, EXP_FOLDER, EXP, "All")

    for set_folder in set_folders:
        set_folder = os.path.basename(set_folder)
        features_set_path = make_path(features_root, set_folder)
        img_types_path = get_subfolders(features_set_path)

        # keep only img_types with "Enhanced" in the name
        img_types_path = [img_type for img_type in img_types_path if "Enhanced" in img_type]

        # keep only None scaler
        scalers = [scaler for scaler in scalers if scaler is None]

        folder_handler = fh.FolderHandler(features_root, EXP_FOLDER, EXP, set_folder)

        if EXP == "Regression":
            regressors = get_regressors()
            results = run_experiment(folder_handler, set_folder, img_types_path, scalers, regressors, EXP,
                                     REGRESSION_TYPE)
            # TODO DELETE save_results_to_latex(results, folder_handler.latex)
            save_best_n_results_to_latex(results, set_folder, folder_handler.latex, n, metric_reg)

        elif EXP == "Classification":
            cross_val = True
            loo = True
            # class_imbalance_handling = 'oversampling'
            class_imbalance_handling = 'none'
            classifiers = get_classifiers()

            results = run_experiment(folder_handler, set_folder, img_types_path, scalers, classifiers, EXP,
                                     cross_val, loo, class_imbalance_handling)
            save_best_n_results_to_latex(results, set_folder, folder_handler.latex, n, metric_cla)
            save_best_n_results_to_latex_by_filters(results, set_folder, folder_handler.latex, n, metric_cla, filters)

        elif EXP == "Anomaly":
            outliers_fraction = 0.15
            cross_val = True
            loo = True

            classifiers_anom = get_anomaly_detectors(outliers_fraction)
            results = run_experiment(folder_handler, set_folder, img_types_path, scalers, classifiers_anom, EXP,
                                     cross_val, loo)
            save_best_n_results_to_latex(results, set_folder, folder_handler.latex, n, metric_ano)
            save_best_n_results_to_latex_by_filters(results, set_folder, folder_handler.latex, n, metric_ano, filters)

        results["Set"] = set_folder
        all_results = pd.concat([all_results, results], ignore_index=False)

    save_best_n_results_to_latex(results, "All", folder_handler_all.latex, n, metric_cla)
    # save_best_n_results_to_latex_by_filters(all_results, "All", folder_handler_all.latex, 1000, metric_cla, ["Set"])
