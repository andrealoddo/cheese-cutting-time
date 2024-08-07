from datetime import datetime

# class imbalance methods
from sklearn.utils import resample

import pandas as pd


def extend_target(data):
    # modify the label of the items of class Negative that are within 15 seconds of the target
    for index, row in data.iterrows():
        if row['label'] == 'Negative':
            # get the timestamp of the item
            timestamp = row['timestamp']
            # get the timestamp of the item with class Positive
            target_timestamp = data[data['label'] == 'Positive']['timestamp'].iloc[0]
            # if the timestamp is within 15 seconds of the target, change the label to Positive
            if target_timestamp - timestamp <= 15 or timestamp - target_timestamp <= 15:
                data.at[index, 'label'] = 'Positive'
    return data


def keep_data_until_target(data):
    # get the timestamp of the item with class Positive
    target_timestamp = data[data['label'] == 'Positive']['timestamp'].iloc[0]

    # keep only the items with timestamp <= target_timestamp
    data = data[data['timestamp'] <= target_timestamp]

    # sort data by timestamp
    data = data.sort_values(by=['timestamp'])

    return data


def exec_normalization(data, scaler):
    # Apply the scaler on the selected columns
    data = scaler.fit_transform(data)

    return data


def get_data_labels_ts(data, *kwargs):
    if kwargs is not None and len(kwargs) > 0 and ('Days' in kwargs[0] or 'Timestamp' in kwargs[0]):
        regression_type = kwargs[0]
    else:
        regression_type = None

    if regression_type == "Days":
        labels = data['label']
        ts = data['label']
        # keep only the numbers from ts
        ts = ts.str.extract('(\d+)').astype(int)
    else:
        labels = data['label']
        # get unique labels
        unique_labels = labels.unique()
        # convert labels to numbers
        labels = labels.replace(unique_labels, range(len(unique_labels)))
        # if timestamp column exist
        if 'timestamp' in data.columns:
            ts = data['timestamp']
        else:
            ts = None

    # drop the columns that exist if they are filenames, labels or timestamps
    features = data.drop(columns=['filename', 'label', 'timestamp'], errors='ignore')

    return features, labels, ts


def encode_datetime_features(timestamps):
    # convert the timestamps to datetime objects
    timestamps = [datetime.strptime(timestamp, '%Y:%m:%d %H:%M:%S') for timestamp in timestamps]

    sorted_timestamps = timestamps.copy()

    # sort the timestamps
    sorted_timestamps.sort()

    # get the minimum timestamp
    min_timestamp = sorted_timestamps[0]

    # set the epoch time to be the minimum timestamp
    epoch_time = min_timestamp

    # encode the timestamps to represent the time difference from the minimum timestamp
    encoded_timestamps = [(timestamp - epoch_time).total_seconds() for timestamp in timestamps]

    return encoded_timestamps


def exec_cleaning(data):
    if 'timestamp' in data.columns:
        # Encode timestamps
        data['timestamp'] = encode_datetime_features(data['timestamp'].to_list())

    return data


def exec_oversampling(data, labels, majority_class, minority_class):
    if labels.nunique():
        print('Error: labels are not binary')
        return data, labels

    # Separate data into majority and minority classes
    majority = data[labels == majority_class]
    minority = data[labels == minority_class]

    # Separate labels into majority and minority classes
    majority_labels = labels[labels == majority_class]
    minority_labels = labels[labels == minority_class]

    # Use the appropriate resampling technique
    oversampled_class = resample(minority, replace=True, n_samples=len(majority), random_state=123)
    oversampled_labels = resample(minority_labels, replace=True, n_samples=len(majority_labels), random_state=123)

    # Combine both classes
    majority = pd.DataFrame(majority)
    oversampled_class = pd.DataFrame(oversampled_class)
    data_oversampled = pd.concat([majority, oversampled_class])

    # Combine both labels
    labels_oversampled = pd.concat([majority_labels, oversampled_labels])

    return data_oversampled, labels_oversampled
