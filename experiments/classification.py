from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from metrics.classification import compute_classification_metrics, compute_classification_metrics_average
from preprocessing.preprocess import exec_normalization, exec_oversampling, get_data_labels_ts


def exec_classification(clf, X_train, y_train, X_test, y_test):

    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_test)

    results = compute_classification_metrics(y_test, y_pred)

    return results


def exec_classification_loo(clf, X_train, y_train, X_test, y_test):

    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_test)

    return y_pred

def perform_classification(data, scaler, clf, cross_val, loo, class_imbalance_handling):

    # Split data into features, labels and timestamps
    data, labels, ts = get_data_labels_ts(data)

    if cross_val:
        results = perform_classification_cross_validation(data, labels, scaler, clf, loo, class_imbalance_handling)
    else:
        results = perform_classification_single_split(data, labels, scaler, clf, class_imbalance_handling)

    if cross_val:
        # perform an average of the results
        results_avg, results_std = compute_classification_metrics_average(results)
    else:
        results_avg, results_std = results, None

    return results_avg, results_std


def perform_classification_cross_validation(data, labels, scaler, alg, loo, class_imbalance_handling):

    if loo:
        kfold = LeaveOneOut()
        y_pred_all = []
        y_test_all = []
    else:
        kfold = KFold(n_splits=5, shuffle=True, random_state=123)

    results = []

    # perform cross validation
    for train_index, test_index in kfold.split(data):
        # Get the training and test data
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        # Data normalization
        if scaler is not None:
            X_train = exec_normalization(X_train, scaler)
            X_test = exec_normalization(X_test, scaler)

        # Class imbalance management
        if class_imbalance_handling == 'oversampling':
            # TODO X_train, y_train = exec_oversampling(X_train, y_train, majority_class, minority_class)
            pass

        if loo:
            y_test_all.append(y_test.iloc[0])
            y_pred = exec_classification_loo(alg, X_train, y_train, X_test, y_test)
            y_pred_all.append(y_pred[0])
        else:
            # Perform classification on the current fold
            fold_results = exec_classification(alg, X_train, y_train, X_test, y_test)
            results.append(fold_results)

    if loo:
        loo_result = compute_classification_metrics(y_test_all, y_pred_all)
        results.append(loo_result)
    else:
        results = results

    return results

def perform_classification_single_split(data, labels, scaler, alg, class_imbalance_handling):

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, random_state=123,
                                                        stratify=labels)
    # Data normalization
    if scaler is not None:
        X_train = exec_normalization(X_train, scaler)
        X_test = exec_normalization(X_test, scaler)

    # Class imbalance management
    if class_imbalance_handling == 'oversampling':
        # TODO X_train, y_train = exec_oversampling(X_train, y_train, majority_class, minority_class)
        pass

    results = exec_classification(alg, X_train, y_train, X_test, y_test)

    return results