import pandas as pd
import os

from results.handling import get_info_from_filename
from utils.datatypes import get_dict_from_classreportstringdict
from utils.files import save_dict_to_csv


def create_results_df(all_sets, task):

    if task == 'AnomalyClassic':
        if all_sets:
            results = pd.DataFrame(columns=['Set', 'ImT', 'Scaler', 'Classifier', 'Features',
                                            'NP', 'NR', 'NF', 'PP', 'PR', 'PF'])
        else:
            results = pd.DataFrame(columns=['ImT', 'Scaler', 'Classifier', 'Features',
                                            'NP', 'NR', 'NF', 'PP', 'PR', 'PF'])
    elif task in ['Classification', 'Anomaly']:
        if all_sets:
            results = pd.DataFrame(columns=['Set', 'ImT', 'Scaler', 'Classifier', 'Features',
                                            'acc', 'Mpr', 'Mre', 'Mf1', 'Msp',
                                            'mpr', 'mre', 'mf1', 'msp',
                                            'wpr', 'wre', 'wf1', 'wsp',
                                            'mcc', 'bacc'])
        else:
            results = pd.DataFrame(columns=['ImT', 'Scaler', 'Classifier', 'Features',
                                            'acc', 'Mpr', 'Mre', 'Mf1',
                                            'mpr', 'mre', 'mf1',
                                            'wpr', 'wre', 'wf1',
                                            'mcc', 'bacc'])
    return results


def add_results_to_df(results_all, filename, task):
    df = pd.read_csv(filename)

    img_type, scaler_name, features, cla_name = get_info_from_filename(filename)

    if task in ['Classification', 'Anomaly']:
        results_all = add_classification_results_to_df(results_all, df, img_type, scaler_name, features, cla_name)
    elif task == 'AnomalyClassic':
        results_all = add_anomaly_results_to_df(results_all, df, img_type, scaler_name, features, cla_name)

    return results_all


def add_anomaly_results_to_df(results_all, df, img_type, scaler_name, features, cla_name):
    negative = get_dict_from_classreportstringdict(df['Negative'][0])
    positive = get_dict_from_classreportstringdict(df['Positive'][0])

    # add the results to the dataframe. Approximate the values to 2 decimal places
    res = pd.DataFrame({'ImT': img_type, 'Scaler': scaler_name, 'Classifier': cla_name, 'Features': features,
                        'NP': "{:.2f}".format(negative['precision']),
                        'NR': "{:.2f}".format(negative['recall']),
                        'NF': "{:.2f}".format(negative['f1-score']),
                        'PP': "{:.2f}".format(positive['precision']),
                        'PR': "{:.2f}".format(positive['recall']),
                        'PF': "{:.2f}".format(positive['f1-score'])},
                       index=[len(results_all)]
                       )

    results_all = results_all.append(res, ignore_index=True)

    return results_all


def add_classification_results_to_df(results_all, df, img_type, scaler_name, features, cla_name):

    macro = get_dict_from_classreportstringdict(df['macro avg'][0])
    micro = get_dict_from_classreportstringdict(df['micro avg'][0])
    weight = get_dict_from_classreportstringdict(df['weighted avg'][0])

    print(df)

    # add the results to the dataframe. Approximate the values to 2 decimal places
    res = pd.DataFrame({'ImT': img_type, 'Scaler': scaler_name, 'Classifier': cla_name, 'Features': features,
                        'acc': "{:.2f}".format(df['accuracy'][0]),
                        'Mpr': "{:.2f}".format(macro['precision']),
                        'Mre': "{:.2f}".format(macro['recall']),
                        'Mf1': "{:.2f}".format(macro['f1-score']),
                        'Msp': "{:.2f}".format(macro['specificity']),
                        'mpr': "{:.2f}".format(micro['precision']),
                        'mre': "{:.2f}".format(micro['recall']),
                        'msp': "{:.2f}".format(micro['specificity']),
                        'mf1': "{:.2f}".format(micro['f1-score']),
                        'wpr': "{:.2f}".format(weight['precision']),
                        'wre': "{:.2f}".format(weight['recall']),
                        'wf1': "{:.2f}".format(weight['f1-score']),
                        'wsp': "{:.2f}".format(weight['specificity']),
                        'mcc': "{:.2f}".format(df['mcc'][0]),
                        'bacc': "{:.2f}".format(df['bacc'][0])},
                       index=[len(results_all)]
                       )

    results_all = results_all.append(res, ignore_index=True)

    return results_all


def save_results_to_disk(results, path):
    save_dict_to_csv([results], path)


def save_plot_to_disk(plot, path):
    plot.savefig(path, bbox_inches='tight')
