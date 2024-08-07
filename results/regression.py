import pandas as pd
import os

from results.handling import get_info_from_filename
from utils.files import save_dict_to_csv


def create_results_df(all_sets):

    if all_sets:
        results = pd.DataFrame(columns=['Set', 'ImT', 'Scaler', 'Regressor', 'Features',
                                        'MSE', 'RMSE', 'MAE', 'R2', 'Adj R2'])
    else:
        results = pd.DataFrame(columns=['ImT', 'Scaler', 'Regressor', 'Features',
                                        'MSE', 'RMSE', 'MAE', 'R2', 'Adj R2'])
    return results


def add_results_to_df(results, filename):
    df = pd.read_csv(filename)

    img_type, scaler_name, features, reg_name = get_info_from_filename(filename)

    # add the results to the dataframe. Approximate the values to 2 decimal places
    res = pd.DataFrame({'ImT': img_type, 'Scaler': scaler_name, 'Regressor': reg_name, 'Features': features,
                        'MSE': "{:.2f}".format(df['mse'][0]),
                        'RMSE': "{:.2f}".format(df['rmse'][0]),
                        'MAE': "{:.2f}".format(df['mae'][0]),
                        'R2': "{:.2f}".format(df['r2'][0]),
                        'Adj R2': "{:.2f}".format(df['adj_r2'][0])},
                       index=[0]
                       )

    results = results.append(res, ignore_index=True)

    return results


def save_results_to_disk(results, path):
    save_dict_to_csv([results], path)


def save_plot_to_disk(plot, path):
    plot.savefig(path, bbox_inches='tight')
