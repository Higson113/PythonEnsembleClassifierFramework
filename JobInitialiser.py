import numpy as np
from datetime import datetime
import warnings
from sklearn.naive_bayes import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

from c45 import C45

from ClassifierLibrary import Adaboost, CostSensitiveAdaBoost, DataBoostIM, PartitionEnsemble
from ClassifierSampleWeightWrapper import ClassifierSampleWeightWrapper
import JobHandler
import DatasetLoader


def initialise_jobs(random_state):
    __jobs = []
    num_boosts = 10

    # __jobs.append() takes a tuple consisting of:
    # STRING: text to use when printing to console,
    # CLASSIFIER: classifier object that implements .fit() and .predict(),
    # PREPROCESSOR: preprocessor object that implements .fit_resample() or use blank string if not needed,
    # COSTSENSITIVE: string name to identify a cost sensitive algorithm, blank if not needed


    c45_resampleable = ClassifierSampleWeightWrapper(C45())
    __jobs.append(("C45 Resampleable:", c45_resampleable, "", ""))

    my_adaboost = Adaboost(base_estimator=c45_resampleable, n_iterations=num_boosts)
    __jobs.append(("My Adaboost, C45 Resampling, Boosts: " + str(num_boosts), my_adaboost, "", ""))

    my_adac1 = CostSensitiveAdaBoost(c45_resampleable, "adac1", num_boosts)
    __jobs.append(("My AdaC1, C45 Resampling, Boosts: " + str(num_boosts), my_adac1, "", "myadac1"))

    my_adac2 = CostSensitiveAdaBoost(c45_resampleable, "adac2", num_boosts)
    __jobs.append(("My AdaC2, C45 Resampling, Boosts: " + str(num_boosts), my_adac2, "", "myadac2"))

    my_adac3 = CostSensitiveAdaBoost(c45_resampleable, "adac3", num_boosts)
    __jobs.append(("My AdaC3, C45 Resampling, Boosts: " + str(num_boosts), my_adac3, "", "myadac3"))

    my_adacost = CostSensitiveAdaBoost(c45_resampleable, "adacost", num_boosts)
    __jobs.append(("My AdaCost, C45 Resampling, Boosts: " + str(num_boosts), my_adacost, "", "myadacost"))


    cart_c45_emulator = DecisionTreeClassifier(max_depth=100, min_samples_leaf=2, min_samples_split=4, criterion='entropy', random_state=random_state)
    minmax_scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
    naive_bayes_comp = ComplementNB()
    #__jobs.append(("CART Classifier", cart_c45_emulator, "", ""))
    # __jobs.append(("Naive Bayes:", naive_bayes_comp, "", ""))

    my_clusterbal = PartitionEnsemble(base_estimator=cart_c45_emulator, data_balancing_strategy="cluster_balance", ensemble_rule="max_dist")
    __jobs.append(("My PartitionEnsemble, ClusterBal + MaxDist , Cart as C4.5 MD100", my_clusterbal, minmax_scaler, ""))

    my_clusterbal = PartitionEnsemble(base_estimator=cart_c45_emulator, data_balancing_strategy="split_balance", ensemble_rule="max_dist")
    __jobs.append(("My PartitionEnsemble, SplitBal + MaxDist, Cart as C4.5 MD100", my_clusterbal, minmax_scaler, ""))

    my_clusterbal = PartitionEnsemble(base_estimator=naive_bayes_comp, data_balancing_strategy="cluster_balance", ensemble_rule="max_dist")
    __jobs.append(("My PartitionEnsemble, ClusterBal + MaxDist, Naive Bayes", my_clusterbal, "", ""))

    my_clusterbal = PartitionEnsemble(base_estimator=naive_bayes_comp, data_balancing_strategy="split_balance", ensemble_rule="max_dist")
    __jobs.append(("My PartitionEnsemble, SplitBal + MaxDist, Naive Bayes", my_clusterbal, "", ""))


    my_databoostim_c45 = DataBoostIM(base_estimator=c45_resampleable, n_iterations=num_boosts)
    __jobs.append(("My DataBoost-IM, C45 Resampleable, Boosts: " + str(num_boosts), my_databoostim_c45, "", ""))


    print("Number of Jobs: " + str(len(__jobs)))
    return __jobs


def main():
    # -----------------------------------EXECUTION--------------------------------------#
    np.seterr(all='raise')
    warnings.filterwarnings("error")
    random_state = np.random.RandomState(0)
    print("Started running at: " + datetime.now().strftime("%H:%M:%S"), end="\r\n")

    forceequalcosts = False
    print("Force Equal Costs is: " + str(forceequalcosts))

    all_datasets = DatasetLoader.load_all_datasets()

    all_jobs = initialise_jobs(random_state)

    # Formatting
    print("")
    for job in all_jobs:
        JobHandler.wrapper(job, all_datasets, printdatasets=True, printdatasetfolds=True, printadametrics=False, forceequalcosts=forceequalcosts)

    print("Finished running at: " + datetime.now().strftime("%H:%M:%S"))
    # -----------------------------------EXECUTION--------------------------------------#

main()
