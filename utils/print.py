from sklearn.metrics import classification_report, accuracy_score
import os


def print_info(folder, scaler_name, alg_name, file_name):
    print(f"Folder: {folder} - Norm: {scaler_name} - Alg: {alg_name} - Feat: {file_name[:-4]}")


def print_regression_results(results):
    # print the results from the dictionary results

    r2_format = "{:.2f}".format(results['r2'])
    adj_r2_format = "{:.2f}".format(results['adj_r2'])
    mae_format = "{:.2f}".format(results['mae'])
    mse_format = "{:.2f}".format(results['mse'])
    rmse_format = "{:.2f}".format(results['rmse'])

    print('Mean Squared Error: ', mse_format)
    print('Root Mean Squared Error: ', rmse_format)
    print('Mean Absolute Error: ', mae_format)
    print('R^2 Score: ', r2_format)
    print('Adjusted R^2 Score: ', adj_r2_format)
    print()


def print_classification_results(y_true, y_pred):
    print(classification_report(y_true, y_pred))

    return accuracy_score(y_true, y_pred)
