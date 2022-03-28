import copy

import numpy as np
from datetime import datetime

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def wrapper(job, datasets, printdatasets=False, printdatasetfolds=False, forceequalcosts=False, printadametrics=False):
    classifier_title, classifier, preprocessor, costsensitive = job

    print(classifier_title + " | Started at " + datetime.now().strftime("%H:%M:%S") )

    # initialising
    clf_roc_auc = []
    clf_accuracy = []
    clf_sensitivity = []
    clf_specificity = []
    clf_gmean = []
    clf_recall = []
    clf_precision = []
    clf_fmeasure = []

    for dataset in datasets:
        dataset_name, samples, classes, ibr_str, n_samples, n_attr, splits = dataset

        if printdatasets:
            print("\t Dataset: " + dataset_name +
                  " | IBR = " + ibr_str +
                  " | Number of Samples = " + str(n_samples) +
                  " | Number of Attributes = " + str(n_attr) +
                  " | K-Folds = " + str(len(splits)) +
                  " | Started at " + datetime.now().strftime("%H:%M:%S")
                  )

        if not printdatasets:
            printdatasetfolds = False
            printadametrics = False

        roc_auc, accuracy, sensitivity, specificity, gmean, recall, precision, fmeasure = stratified_kfold_runner(samples, classes, preprocessor, costsensitive, splits, classifier, dataset_name, printmetrics=printdatasets, printfoldmetrics=printdatasetfolds, forceequalcosts=forceequalcosts, printadametrics=printadametrics)

        clf_roc_auc.append(roc_auc)

        clf_accuracy.append(accuracy)
        clf_sensitivity.append(sensitivity)
        clf_specificity.append(specificity)
        clf_gmean.append(gmean)

        clf_recall.append(recall)
        clf_precision.append(precision)
        clf_fmeasure.append(fmeasure)

    # Printing classifier metrics
    print("Metrics for " + classifier_title)
    print("Mean ROC AUC: " + "{:.6f}".format(np.sum(clf_roc_auc) / len(clf_roc_auc)), end="")
    print(" | Mean Gmean: " + "{:.6f}".format(np.sum(clf_gmean) / len(clf_gmean)), end="")
    print(" | Mean Test Accuracy: " + "{:.6f}".format(np.sum(clf_accuracy) / len(clf_accuracy)), end="")
    print(" | Mean Test Sensitivity: " + "{:.6f}".format(np.sum(clf_sensitivity) / len(clf_sensitivity)), end="")
    print(" | Mean Test Specificity: " + "{:.6f}".format(np.sum(clf_specificity) / len(clf_specificity)), end="")
    print(" | Mean Test Fmeasure: " + "{:.6f}".format(np.sum(clf_fmeasure) / len(clf_fmeasure)), end="")
    print(" | Mean Test Recall: " + "{:.6f}".format(np.sum(clf_recall) / len(clf_recall)), end="")
    print(" | Mean Test Precision: " + "{:.6f}".format(np.sum(clf_precision) / len(clf_precision)))
    print()
    print()


def stratified_kfold_runner(__samples, __classes, preprocessor, costsensitive, splits, passed_clf, dataset_name,  printfoldprogress=False, printfoldmetrics=False, printmetrics=False, printadametrics=False, forceequalcosts=False):
    kfold_split = 1
    dataset_sensitivities = []  # TP Rate (Minority)
    dataset_specificities = []  # TN Rate (Majority)
    dataset_accuracies = []
    dataset_roc_auc = []
    dataset_gmean = []

    dataset_recall = []
    dataset_precision = []
    dataset_fmeasure = []

    for train, test in splits:
        if printfoldprogress:
            print("KF " + str(kfold_split) + " | ", end="")
        clf = copy.deepcopy(passed_clf)
        # training set
        training_samples = __samples[train, :]
        training_classes = __classes[train]
        # testing set
        test_samples = __samples[test, :]
        test_true_classes = __classes[test]

        # running preprocessor if included
        if preprocessor:
            if isinstance(preprocessor, MinMaxScaler):
                fitted_minmaxscaler = preprocessor.fit(training_samples)
                training_samples = fitted_minmaxscaler.transform(training_samples)
                test_samples = fitted_minmaxscaler.transform(test_samples)
            elif isinstance(preprocessor, StandardScaler):
                training_samples = preprocessor.fit_transform(training_samples)
                test_samples = preprocessor.transform(test_samples)
            else:
                training_samples, training_classes = preprocessor.fit_resample(training_samples, training_classes)

        if costsensitive:
            # Initialising costs
            costs = np.copy(training_classes).astype(float)
            # cost is 1 for minority class
            costs[costs == 1] = 1.

            # Set costs manually to replicate experiments from Sun et al 2007
            if dataset_name in ("Breast Cancer Yugoslavia", "Hepatitis", "Pima Diabetes", "sick_euthyroid"):
                if costsensitive == "myadac1":
                    experiment_cost = {"Breast Cancer Yugoslavia": 0.9, "Hepatitis": 0.1, "Pima Diabetes": 0.6, "sick_euthyroid": 0.5}
                    maj_cost = experiment_cost[dataset_name]
                elif costsensitive == "myadac2":
                    experiment_cost = {"Breast Cancer Yugoslavia": 0.6, "Hepatitis": 0.5, "Pima Diabetes": 0.9, "sick_euthyroid": 0.9}
                    maj_cost = experiment_cost[dataset_name]
                elif costsensitive == "myadac3":
                    experiment_cost = {"Breast Cancer Yugoslavia": 0.6, "Hepatitis": 0.8, "Pima Diabetes": 0.9, "sick_euthyroid": 0.9}
                    maj_cost = experiment_cost[dataset_name]
                elif costsensitive == "myadacost":
                    experiment_cost = {"Breast Cancer Yugoslavia": 0.4, "Hepatitis": 0.8, "Pima Diabetes": 0.3, "sick_euthyroid": 0.7}
                    maj_cost = experiment_cost[dataset_name]
            else:
                # cost is based on 1/IBR for majority class, with floor of 0.001 to prevent maths errors
                maj_cost = sum(training_classes == 1) / sum(training_classes == -1)
                if maj_cost < 1e-3:
                    maj_cost = 0.001

            costs[costs == -1] = maj_cost

            # DEBUGGING IF ADACx reduces to AdaBoost when costs are equal
            if forceequalcosts:
                costs[:] = 1.
                maj_cost = 1.

            algs = {"myadac1", "myadac2", "myadac3", "myadacost"}
            if costsensitive in algs:
                clf.fit(training_samples, training_classes, costs)
            else:
                raise ValueError("costsensitive identifier not set for this algorithm, cannot call .fit()")
        else:
            clf.fit(training_samples, training_classes)

        if printadametrics and (hasattr(clf, 'ensemble') or hasattr(clf, 'estimator_weights_') or hasattr(clf, 'alphas_') or hasattr(clf, 'clf_weights') or hasattr(clf, 'estimator_alphas_')):
            print("\t\t\tWeights - ", end="")
            try:
                for model, weight, error in clf.ensemble:
                    print("A:" + "{:.10f}".format(weight) + " | ", end="")
            except:
                pass

            print("\t\t\tErrors  - ", end="")
            try:
                for model, weight, error in clf.ensemble:
                    print("E:" + "{:.10f}".format(error) + " | ", end="")
            except:
                pass

        test_predicted_classes = clf.predict(test_samples)

        if test_predicted_classes is None:
            test_predicted_classes = np.zeros(len(test_samples), dtype=np.float64)

        # if the classifier gives a predicted class label of 0, the classifier failed for that example.
        if 0 in test_predicted_classes:
            print("\t\t\t Classifier predicted sample as 0 when classes must be {-1,1}, setting to be incorrect (-y)")
            # To prevent crashing, force it to be classed as the incorrect class for stats calculation
            test_predicted_classes[test_predicted_classes == 0] = -test_true_classes[test_predicted_classes == 0]

        if len(test_predicted_classes) > 0:
            # deriving metrics
            split_roc_auc = roc_auc_score(test_true_classes, test_predicted_classes)
            dataset_roc_auc.append(split_roc_auc)
            split_accuracy = accuracy_score(test_true_classes, test_predicted_classes)
            dataset_accuracies.append(split_accuracy)
            split_sensitivity = sensitivity_score(test_true_classes, test_predicted_classes, average='binary', pos_label=1)
            dataset_sensitivities.append(split_sensitivity)
            split_specificity = specificity_score(test_true_classes, test_predicted_classes, average='binary', pos_label=1)
            dataset_specificities.append(split_specificity)
            split_gmean = geometric_mean_score(test_true_classes, test_predicted_classes, average='binary', pos_label=1)
            dataset_gmean.append(split_gmean)

            split_recall = recall_score(test_true_classes, test_predicted_classes, average='binary', pos_label=1)
            dataset_recall.append(split_recall)
            split_precision = precision_score(test_true_classes, test_predicted_classes, average='binary', pos_label=1, zero_division=0)
            dataset_precision.append(split_precision)
            split_fmeasure = f1_score(test_true_classes, test_predicted_classes, average='binary', pos_label=1)
            dataset_fmeasure.append(split_fmeasure)

            if printfoldmetrics:
                print("\t \t S" + "{:02d}".format(kfold_split) + " Test ROC AUC: " + "{:.6f}".format(split_roc_auc), end="")
                print(" | S" + "{:02d}".format(kfold_split) + " Test Gmean: " + "{:.6f}".format(split_gmean), end="")
                print(" -- S" + "{:02d}".format(kfold_split) + " Test Acc: " + "{:.6f}".format(split_accuracy), end="")
                print(" | S" + "{:02d}".format(kfold_split) + " Test Sens: " + "{:.6f}".format(split_sensitivity), end="")
                print(" | S" + "{:02d}".format(kfold_split) + " Test Spec: " + "{:.6f}".format(split_specificity), end="")
                print(" -- S" + "{:02d}".format(kfold_split) + " Test FMeas: " + "{:.6f}".format(split_fmeasure), end="")
                print(" | S" + "{:02d}".format(kfold_split) + " Test Rec: " + "{:.6f}".format(split_recall), end="")
                print(" | S" + "{:02d}".format(kfold_split) + " Test Prec: " + "{:.6f}".format(split_precision))

        kfold_split = kfold_split + 1

    dataset_mean_roc_auc = np.sum(dataset_roc_auc) / len(dataset_roc_auc)

    dataset_mean_accuracy = np.sum(dataset_accuracies) / len(dataset_accuracies)
    dataset_mean_sensitivity = np.sum(dataset_sensitivities) / len(dataset_sensitivities)
    dataset_mean_specificity = np.sum(dataset_specificities) / len(dataset_specificities)
    dataset_mean_gmean = np.sum(dataset_gmean) / len(dataset_gmean)

    dataset_mean_recall = np.sum(dataset_recall) / len(dataset_recall)
    dataset_mean_precision = np.sum(dataset_precision) / len(dataset_precision)
    dataset_mean_fmeasure = np.sum(dataset_fmeasure) / len(dataset_fmeasure)

    if printmetrics:
        print("\t Mean Dataset ROC AUC: " + "{:.6f}".format(dataset_mean_roc_auc), end="")
        print(" | Mean Dataset Test Gmean: " + "{:.6f}".format(dataset_mean_gmean), end="")
        print(" -- Mean Dataset Test Acc: " + "{:.6f}".format(dataset_mean_accuracy), end="")
        print(" | Mean Dataset Test Sens: " + "{:.6f}".format(dataset_mean_sensitivity), end="")
        print(" | Mean Dataset Test Spec: " + "{:.6f}".format(dataset_mean_specificity), end="")
        print(" -- Mean Dataset Test FMeas: " + "{:.6f}".format(dataset_mean_fmeasure), end="")
        print(" | Mean Dataset Test Recall: " + "{:.6f}".format(dataset_mean_recall), end="")
        print(" | Mean Dataset Test Precision: " + "{:.6f}".format(dataset_mean_precision))
        print("\t .")

    return dataset_mean_roc_auc, dataset_mean_accuracy, dataset_mean_sensitivity, dataset_mean_specificity, dataset_mean_gmean, dataset_mean_recall, dataset_mean_precision, dataset_mean_fmeasure
