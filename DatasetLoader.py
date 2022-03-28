from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from imblearn import datasets as imbalanced_datasets
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.datasets import fetch_openml
import numpy as np


def load_all_datasets():
    kfolds = 5
    print("Number of K-Folds in Stratified Cross Validation: " + str(kfolds))

    filter_imblearn_keel = (["ecoli", "yeast_me2", "abalone_19"])
    filter_imblearn_adacx = (["sick_euthyroid"])
    filter_imblearn_databoostim = (["oil", "satimage","thyroid_sick"])
    filter_imblearn_clusterbal = (["ecoli", "yeast_me2"])

    data = []
    data = load_imblearn_datasets(kfolds, filter_imblearn_keel)
    data = data + load_openml_datasets(kfolds, useAdaCx=False, useDBIM=False, useClusterBal=False)

    print("Number of Datasets: " + str(len(data)))
    return data


def test_train_creator(samples, classes, kfolds):
    # DataBoost-IM Paper: 10-fold, 5 Repeats
    # ClusterBal Paper: 10-fold, 10 Repeats
    n_repeats = 5
    rskfcv = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=n_repeats, random_state=0)

    # Cost-Sensitive AdaBoost Paper: 10 folds, 80/20 split
    ss = ShuffleSplit(kfolds, test_size=0.2, train_size=0.8)

    splits = []
    generator = rskfcv.split(samples, classes)
    for train, test in generator:
        splits.append((train, test))

    return splits


def load_imblearn_datasets(kfolds, filter_data):
    # Download location for imblearn datasets. Remove 'data_home=' parameter from fetch_datasets() to use default location
    imblearn_data_home = "C:\\Users\\mikeh\\Actual Documents\\Loughborough University Studies\\FYP ECIP Coding\\Python\\imblearn_datasets"

    __data = []

    # Reading imbalanced dataset repo from sklearn extension imbalanced_learn
    # param for selecting specific dataset..    filter_data=([2, 1])      filter_data=(["abalone","ecoli"])
    # Filter data for KEEL datasets     filter_data=(["ecoli","yeast_me2","abalone_19"])        "libras_move"
    imb_data_repo = imbalanced_datasets.fetch_datasets(data_home=imblearn_data_home, verbose=True, filter_data=filter_data)

    # remaking object for ease of use
    for dataset_name in imb_data_repo:
        samples = imb_data_repo[dataset_name].data
        classes = imb_data_repo[dataset_name].target

        splits = test_train_creator(samples, classes, kfolds)
        ibr_string = "{:.2f}".format(sum(classes == -1) / sum(classes == 1))
        n_samples = len(classes)
        n_attr = len(samples[1, :])
        __data.append((dataset_name, samples, classes, ibr_string, n_samples, n_attr, splits))
        print(dataset_name + ", ", end="")

    return __data


def load_openml_datasets(kfolds, useAdaCx, useDBIM, useClusterBal):
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent", add_indicator=False)
    iterative_imputer = IterativeImputer(missing_values=np.nan, max_iter=30, min_value=0)

    # Download location for imblearn datasets. Remove 'data_home=' parameter from fetch_datasets() to use default location
    openml_data_home = "C:\\Users\\mikeh\\Actual Documents\\Loughborough University Studies\\FYP ECIP Coding\\Python\\openml_datasets"

    __data = []

    if useAdaCx:
        # AdaCx Dataset: Breast Cancer Yugoslavia dataset
        __data.append(openml_dataset_retriever(openml_data_home, 13, "Breast Cancer Yugoslavia", kfolds, min_label="recurrence-events", maj_label="no-recurrence-events", imputer=simple_imputer))
        # AdaCx Dataset: Hepatitis dataset
        __data.append(openml_dataset_retriever(openml_data_home, 55, "Hepatitis", kfolds, min_label="DIE", maj_label="LIVE", imputer=simple_imputer))
        # AdaCx Dataset: Pima Diabetes Dataset
        __data.append(openml_dataset_retriever(openml_data_home, 37, "Pima Diabetes", kfolds, min_label="tested_positive", maj_label="tested_negative", imputer=simple_imputer))

    if useDBIM:
        # DataBoost-IM Datasets: Sonar
        __data.append(openml_dataset_retriever(openml_data_home, 40, "Sonar", kfolds, min_label="Rock", maj_label="Mine", imputer=simple_imputer))
        # DataBoost-IM Datasets: ionosphere
        __data.append(openml_dataset_retriever(openml_data_home, 59, "ionosphere", kfolds, min_label="b", maj_label="g", imputer=simple_imputer))
        # DataBoost-IM Datasets: Wisconcin Breast Cancer
        __data.append(openml_dataset_retriever(openml_data_home, 15, "Breast Cancer Wisconsin", kfolds, min_label="malignant", maj_label="benign", imputer=simple_imputer))
        # DataBoost-IM Datasets: Breast Cancer Yugoslavia dataset
        __data.append(openml_dataset_retriever(openml_data_home, 13, "Breast Cancer Yugoslavia", kfolds, min_label="recurrence-events", maj_label="no-recurrence-events", imputer=simple_imputer))
        # DataBoost-IM Datasets: phoneme
        __data.append(openml_dataset_retriever(openml_data_home, 1489, "phoneme", kfolds, min_label='2', maj_label='1', imputer=simple_imputer))
        # DataBoost-IM Datasets: vehicle
        __data.append(openml_dataset_retriever(openml_data_home, 54, "vehicle", kfolds, min_label='van', imputer=simple_imputer))
        # DataBoost-IM Datasets: Hepatitis dataset
        __data.append(openml_dataset_retriever(openml_data_home, 55, "Hepatitis", kfolds, min_label="DIE", maj_label="LIVE", imputer=simple_imputer))
        # DataBoost-IM Datasets: Segment (UNSURE)
        __data.append(openml_dataset_retriever(openml_data_home, 958, "segment", kfolds, min_label='P', maj_label='N', imputer=simple_imputer))
        # DataBoost-IM Datasets: Glass
        __data.append(openml_dataset_retriever(openml_data_home, 41, "glass", kfolds, min_label='headlamps', imputer=simple_imputer))
        # DataBoost-IM Datasets: Vowel (EQUAL NUMBER OF CLASS VALUES, GUESSING 1st class hid)
        __data.append(openml_dataset_retriever(openml_data_home, 307, "vowel", kfolds, min_label='hid', imputer=simple_imputer))
        # DataBoost-IM Datasets: Abalone9-18
        __data.append(openml_dataset_retriever(openml_data_home, 183, "abalone min=18, maj=9", kfolds, min_label='18', maj_label="9", imputer=simple_imputer))
        # DataBoost-IM Datasets: Yeast CYT vs POX
        __data.append(openml_dataset_retriever(openml_data_home, 181, "yeast min=pox, maj=cyt", kfolds, min_label='POX', maj_label="CYT", imputer=simple_imputer))
        # DataBoost-IM Datasets: Primary Tumor - Thyroid vs Rest  (might be coln instead?)
        __data.append(openml_dataset_retriever(openml_data_home, 171, "primary tumor", kfolds, min_label='thyroid', imputer=simple_imputer))

    if useClusterBal:
        # ClusterBal Dataset: Yeast3
        __data.append(openml_dataset_retriever(openml_data_home, 181, "Yeast3", kfolds, min_label="ME3"))
        # ClusterBal Dataset: Yeast5
        __data.append(openml_dataset_retriever(openml_data_home, 181, "Yeast5", kfolds, min_label="ME1"))
        # ClusterBal Dataset: Yeast6
        __data.append(openml_dataset_retriever(openml_data_home, 181, "Yeast6", kfolds, min_label="EXC"))

    return __data


def openml_dataset_retriever(openml_data_home, data_id, dataset_name, kfolds, min_label=None, maj_label=None, imputer=None, target_column=None):
    if target_column is None:
        dataset = fetch_openml(data_id=data_id, data_home=openml_data_home, return_X_y=True, as_frame=False)
    else:
        dataset = fetch_openml(data_id=data_id, data_home=openml_data_home, return_X_y=True, as_frame=False, target_column=target_column)

    samples = dataset[0]
    if imputer is not None:
        samples = imputer.fit_transform(samples)

    classes = dataset[1]
    if maj_label is None and min_label is not None:  # Specifying only min class
        if sum(classes != min_label) == 0 or sum(classes == min_label) == 0:
            raise ValueError("error relabelling maj/min classes when retrieving from openml, datasetid:" + str(data_id))
        classes[classes != min_label] = -1
        classes[classes == min_label] = 1
    elif maj_label is not None and min_label is not None:  # Specifying both maj and min class
        if sum(classes == maj_label) == 0 or sum(classes == min_label) == 0:
            raise ValueError("error relabelling maj/min classes when retrieving from openml, datasetid:" + str(data_id))
        boolean_indicies = np.logical_and(classes != maj_label, classes != min_label)
        classes = np.delete(classes, boolean_indicies, axis=0)
        samples = np.delete(samples, boolean_indicies, axis=0)
        classes[classes == maj_label] = -1
        classes[classes == min_label] = 1
    elif maj_label is not None and min_label is None:  # Specifying only maj class
        if sum(classes != maj_label) == 0 or sum(classes == maj_label) == 0:
            raise ValueError("error relabelling maj/min classes when retrieving from openml, datasetid:" + str(data_id))
        classes[classes != maj_label] = 1
        classes[classes == maj_label] = -1

    classes = np.asarray(classes, dtype=np.int64)
    ibr_string = "{:.2f}".format(sum(classes == -1) / sum(classes == 1))

    splits = test_train_creator(samples, classes, kfolds)

    n_samples = len(classes)
    n_attr = len(samples[1, :])
    print(dataset_name + ", ", end="")
    return dataset_name, samples, classes, ibr_string, n_samples, n_attr, splits
