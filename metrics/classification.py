from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    balanced_accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report
import copy
import numpy as np


def compute_classification_metrics(y_true, y_pred):
    # create a dictionary to store the results, with the following keys:
    class_report_dict = {}

    # if the confusion matrix is not a 2x2 matrix, print a warning
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) != 2:
        # create a int 2x2 matrix
        cm_temp = np.zeros((2, 2), dtype=int)
        if max(y_true) == 0:
            cm_temp[0, 0] = cm[0, 0]
            cm = cm_temp
        elif max(y_true) == 1:
            cm_temp[1, 1] = cm[0, 0]
            cm = cm_temp
        else:
            print("Warning: The confusion matrix is not a 2x2 matrix and we can't handle it. ")

    class_report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    class_report_dict['mcc'] = matthews_corrcoef(y_true, y_pred)
    class_report_dict['bacc'] = balanced_accuracy_score(y_true, y_pred)

    # add micro average to class_report_dict
    class_report_dict['micro avg'] = {}
    class_report_dict['micro avg']['precision'] = precision_score(y_true, y_pred, average='micro')
    class_report_dict['micro avg']['recall'] = recall_score(y_true, y_pred, average='micro')
    class_report_dict['micro avg']['f1-score'] = f1_score(y_true, y_pred, average='micro')
    class_report_dict['micro avg']['specificity'] = recall_score(y_true, y_pred, average='micro', pos_label=0)
    # get the support from the macro average
    class_report_dict['micro avg']['support'] = class_report_dict['macro avg']['support']


    class_report_dict['macro avg']['specificity'] = recall_score(y_true, y_pred, average='macro', pos_label=0)
    class_report_dict['weighted avg']['specificity'] = recall_score(y_true, y_pred, average='weighted', pos_label=0)

    # add confusion matrix to class_report_dict
    class_report_dict['confusion_matrix'] = cm

    # roc_auc = roc_auc_score(y_true, y_pred, average='weighted')
    # pr_auc = average_precision_score(y_true, y_pred, average='weighted')

    return class_report_dict


def compute_classification_metrics_average(results_or):
    results_for_avg = copy.deepcopy(results_or)
    results_for_std = copy.deepcopy(results_or)

    # create empty dictionaries to store the sum, average and standard deviation of the results
    results_sum = {}
    results_avg = {}
    results_std = {}

    for r in results_for_avg:
        for k, v in r.items():
            if k not in results_sum:
                results_sum[k] = v
            else:
                if isinstance(v, dict):
                    for k1, v1 in v.items():
                        results_sum[k][k1] += v1
                else:
                    results_sum[k] += v

    # compute the average results
    for k, v in results_sum.items():
        if isinstance(v, dict):
            results_avg[k] = {}
            for k1, v1 in v.items():
                results_avg[k][k1] = v1 / len(results_for_avg)
        else:
            results_avg[k] = v / len(results_for_avg)

    # compute the standard deviation of the results
    for r in results_for_std:
        for k, v in r.items():
            if k not in results_std:
                results_std[k] = v
            else:
                if isinstance(v, dict):
                    for k1, v1 in v.items():
                        results_std[k][k1] = (v1 - results_avg[k][k1]) ** 2
                else:
                    results_std[k] = (v - results_avg[k]) ** 2

    for k, v in results_std.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                results_std[k][k1] = (v1 / len(results_for_avg)) ** 0.5
        else:
            results_std[k] = (v / len(results_for_avg)) ** 0.5

    return results_avg, results_std
