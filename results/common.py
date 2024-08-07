import pandas as pd
import results.regression as reg
import results.classification as cla


def create_empty_df_from_task(task, *kwargs):
    results = pd.DataFrame()

    if kwargs:
        all_sets = True
    else:
        all_sets = False

    if task == 'Regression':
        results = reg.create_results_df(all_sets)
    else: # task in ['Classification', 'Anomaly']:
        results = cla.create_results_df(all_sets, task)

    return results


def add_results_to_df_by_task(task, results, results_path):
    if task == 'Regression':
        results = reg.add_results_to_df(results, results_path)
    else:   # if task in ['Classification', 'Anomaly']:
        results = cla.add_results_to_df(results, results_path, task)

    return results
