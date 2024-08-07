import pandas as pd

from utils.folders import make_path


def make_latex_table_from_df(df):
    # check if df is a list of dataframes
    if isinstance(df, list):
        temp_df = df[0]
    else:
        temp_df = df

    # define the table caption
    img_type = temp_df['ImT']
    caption = 'ImT = ' + img_type + '; Scaler = ' + temp_df['Scaler']
    label = 'tab:' + img_type + '-' + temp_df['Scaler']

    # define the string formatting for numerical columns
    num_format = '{:.2f}'

    # define the table headers
    headers = ['Regressor', 'Features', 'MSE', 'RMSE', 'MAE', 'R2', 'Adj R2']

    # create the LaTeX table
    table = '\\begin{table}[h]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{1pt}\n\\caption{' + caption + \
            '}\\label{' + label + '}\n\\begin{tabular}{ll ccccc}\n\\toprule\n'

    # add the table headers
    for header in headers:
        table += '\\textbf{' + header + '} & '
    table = table[:-2] + '\\\\\\midrule\n'

    # add the table rows
    for index, row in df.iterrows():

        if row['Features'] == "concatenated_data_all":
            row_features = 'merge'
        else:
            row_features = row['Features']
        r = row['Regressor'] + ' & ' + row_features + ' & '

        # cycle through the numerical columns
        for i, col in enumerate(row[4:]):

            col_num = float(col)
            if col_num > 500 or col_num < -500:
                r = r + ' - '
            else:
                # check if col_num is equal to the minimum value of that column
                if col_num == df.iloc[:, i + 4].astype('float64').min():
                    r = r + '\\textbf{' + num_format.format(float(col)) + '}'
                else:
                    r = r + num_format.format(float(col))

            # check if col is the last column
            if i == len(row[4:]) - 1:
                r = r + ' \\\\\n'
            else:
                r = r + ' & '

        table += r

        # complete the LaTeX table
    table += '\\bottomrule\n\\end{tabular}\n\\end{table}'

    # replace every '_' with '\_' to avoid LaTeX errors
    table = table.replace('_', '\_')

    return table


def save_table_to_latex(table, set, folder, filename, modality='w'):
    file = make_path(folder, filename + '.tex')
    with open(file, modality) as f:
        f.write(table)
        f.write('\n\n')


def save_results_to_latex(results, folder):
    for result in results:

        img_type = result['ImT'][0]

        # Get all the feature types
        feature_types = result['Features'].unique()

        for feat in feature_types:
            # find the first and last index of the feature type
            first_index = result[result['Features'] == feat].index[0]
            last_index = result[result['Features'] == feat].index[-1]

            # create a new dataframe with the results of the feature type and without index
            temp_result = result.iloc[first_index:last_index + 1, :].reset_index(drop=True)

            table = make_latex_table_from_df(temp_result)

            save_table_to_latex(table, folder, img_type, 'a')


def get_columns_conf_from_first_row(first_row):
    columns_conf = ''
    for item in first_row:
        try:
            if isinstance(float(item), float):
                columns_conf += 'c'
            else:
                columns_conf += 'l'
        except ValueError:
            columns_conf += 'l'

    return columns_conf


def make_latex_table_from_single_df(df, caption):
    # define the string formatting for numerical columns
    num_format = '{:.2f}'

    # get the headers from the dataframe
    headers = df.columns.values.tolist()
    columns_conf = get_columns_conf_from_first_row(df.iloc[0, :].values.tolist())
    label = 'tab:' + caption.replace('_', '-')

    # create the LaTeX table
    table = '\\begin{table}[h]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{1.2pt}\n\\caption{' + caption + \
            '}\\label{' + label + '}\n\\begin{tabular}{' + columns_conf + '}\n\\toprule\n'

    # add the table headers
    for header in headers:
        table += '\\textbf{' + header + '} & '
    table = table[:-2] + '\\\\\\midrule\n'

    # add the table rows
    for index, row in df.iterrows():

        # add the row to the table
        for item in row:
            # if the item is a float, format it
            if isinstance(item, float):
                table += num_format.format(item) + ' & '
            else:
                table += str(item) + ' & '

        table = table[:-2] + '\\\\\n'

    # complete the LaTeX table
    table += '\\bottomrule\n\\end{tabular}\n\\end{table}'

    # replace every '_' with '\_' to avoid LaTeX errors
    table = table.replace('_', '\_')

    return table


def save_best_n_results_to_latex(results, set_f, folder, n, metric, *kwargs):
    if kwargs:
        filename = kwargs[0]
    else:
        filename = set_f + '_best_' + str(n) + '_results_' + metric

    # merge all the results into one dataframe
    # results = pd.concat(results, ignore_index=True)

    # convert the column of that metric to positive values. Avoid SettingWithCopyWarning
    results = results.copy()
    results[metric] = results[[metric]].astype(float).abs()

    # keep only the results with NF column >= 0.6 and PF column >= 0.6. Convert the columns to float
    if 'NF' in results.columns:
        results = results[(results['NF'].astype(float) >= 0.1) & (results['PF'].astype(float) >= 0.1)]
    elif 'RMSE' in results.columns:
        results = results[(results['RMSE'].astype(float) < 200)]
    elif 'acc' in results.columns:
        results = results[(results['acc'].astype(float) > 0.1)]


    if results.empty:
        return
    # else, if the number of results is less than n, set n to the number of results
    if len(results) < n:
        n = len(results)

    # sort the results by the metric
    results = results.sort_values(by=metric, ascending=False)

    # get the best n results
    results = results.iloc[:n, :]

    table = make_latex_table_from_single_df(results, filename)

    save_table_to_latex(table, set_f, folder, filename, 'w')


def save_best_n_results_to_latex_by_filters(results, set_f, folder, n, metric, filters):
    for filt in filters:

        if "Set" in filt:
            all_results_filt = pd.DataFrame()
            # get unique sets
            sets = results["Set"].unique()
            for set in sets:
                results_filt = results[results["Set"] == set]
                all_results_filt = pd.concat([all_results_filt, results_filt], ignore_index=False)
            filename = set_f + '_best_' + str(n) + '_results_' + metric + '_filter_' + 'all_sets'
            save_best_n_results_to_latex(all_results_filt, set_f, folder, n, metric, filename)

        if "ImT" in filt:
            # get unique image types
            img_types = results["ImT"].unique()
            for img_type in img_types:
                results_filt = results[results["ImT"] == img_type]
                filename = set_f + '_best_' + str(n) + '_results_' + metric + '_filter_' + img_type
                save_best_n_results_to_latex(results_filt, set_f, folder, n, metric, filename)

        if "Scaler" in filt:
            # get unique scaler types
            scaler_types = results["Scaler"].unique()
            for scaler_type in scaler_types:
                results_filt = results[results["Scaler"] == scaler_type]
                filename = set_f + '_best_' + str(n) + '_results_' + metric + '_filter_' + scaler_type
                save_best_n_results_to_latex(results_filt, set_f, folder, n, metric, filename)

        if "Classifier" in filt:
            # get unique classifier types
            classifier_types = results["Classifier"].unique()
            for classifier_type in classifier_types:
                results_filt = results[results["Classifier"] == classifier_type]
                filename = set_f + '_best_' + str(n) + '_results_' + metric + '_filter_' + classifier_type
                save_best_n_results_to_latex(results_filt, set_f, folder, n, metric, filename)

        if "Features" in filt:
            # get unique feature types
            feature_types = results["Features"].unique()
            for feature_type in feature_types:
                results_filt = results[results["Features"] == feature_type]
                filename = set_f + '_best_' + str(n) + '_results_' + metric + '_filter_' + feature_type
                save_best_n_results_to_latex(results_filt, set_f, folder, n, metric, filename)
