import numpy as np
import copy
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist
import math
from imblearn.metrics import sensitivity_score, specificity_score


class PartitionEnsemble:
    def __init__(self, base_estimator, data_balancing_strategy, ensemble_rule):
        self.base_estimator = base_estimator
        self.data_balancing_strategy = data_balancing_strategy
        self.ensemble_rule = ensemble_rule
        self.ensemble = []

    # samples: List of training samples
    # classes: List of true sample classes {-1, 1}
    def fit(self, samples, classes):
        # calculate required number of subsets K
        majority_samples = samples[classes == -1]
        minority_samples = samples[classes == 1]
        K = math.ceil(len(majority_samples) / len(minority_samples))

        subsets = []
        if self.data_balancing_strategy == "cluster_balance":
            # split majority class samples into K clusters
            kmean_clustering = k_means(X=majority_samples, n_clusters=K, algorithm='full', init='random', n_init=10, random_state=0)
            # get clusters indicies for each sample
            cluster_idxs = kmean_clustering[1]

            # for each majority cluster
            for i in range(K):
                # gather majority samples in ith cluster
                subset_maj_samples = majority_samples[cluster_idxs == i]
                # combine majority cluster with a copy of minority class to form a subset
                subset_samples = np.append(subset_maj_samples, minority_samples, axis=0)

                # create list indexing each sample's class (-1 for maj, +1 for minority)
                subset_maj_classes = -np.ones(len(subset_maj_samples))
                subset_classes = np.append(subset_maj_classes, np.ones(len(minority_samples)), axis=0)

                # append subset to the list of subsets
                subsets.append((subset_samples, subset_classes))

        elif self.data_balancing_strategy == "split_balance":
            # split majority class samples into K random partitions
            np.random.shuffle(majority_samples)
            maj_partitions = np.array_split(ary=majority_samples, indices_or_sections=K, axis=0)

            # for each majority partition
            for maj_partition in maj_partitions:
                # combine majority partition with a copy of minority class into a subset
                subset_samples = np.append(maj_partition, minority_samples, axis=0)

                # create list indexing each sample's class (-1 for maj, +1 for minority)
                subset_maj_classes = -np.ones(len(maj_partition))
                subset_classes = np.append(subset_maj_classes, np.ones(len(minority_samples)), axis=0)

                # append subset to the list of subsets
                subsets.append((subset_samples, subset_classes))
        else:
            raise ValueError("Invalid data_balancing_strategy passed during instantiation")

        # train an individual base classifier on each subset to form an ensemble
        for subset_samples, subset_classes in subsets:
            clf = copy.deepcopy(self.base_estimator)
            clf.fit(subset_samples, subset_classes)
            self.ensemble.append((clf, subset_samples, subset_classes))

    # samples: List of samples to predict
    def predict(self, samples):
        if self.ensemble_rule == "max_dist":
            R1 = None
            R2 = None

            # for each classifier and subset i
            for clf, subset_samples, subset_classes in self.ensemble:
                # calculating P(i,1) and P(i,2)
                clf_probabilities = clf.predict_proba(samples)

                clf_avg_dist = []
                for sample in samples:
                    sample = np.asarray([sample])  # cdist requires sample to be a 2d array

                    # calculating D(i,1)
                    minority_samples = subset_samples[subset_classes == 1]
                    minority_distances = cdist(sample, minority_samples, metric="euclidean")[0]
                    average_minority_distance = minority_distances.mean()

                    # calculating D(i,2)
                    majority_samples = subset_samples[subset_classes == -1]
                    majority_distances = cdist(sample, majority_samples, metric="euclidean")[0]
                    average_majority_distance = majority_distances.mean()

                    # Recording D(i,j)
                    clf_avg_dist.append([average_majority_distance, average_minority_distance])

                clf_avg_dist = np.asarray(clf_avg_dist)
                # Pi1 / (Di1 + 1)
                R1_temp = clf_probabilities[:, 1] / (clf_avg_dist[:, 1] + 1)
                # Pi2 / (Di2 + 1)
                R2_temp = clf_probabilities[:, 0] / (clf_avg_dist[:, 0] + 1)

                if R1 is None:
                    R1 = R1_temp  # if 1st classifier, assign all vals to R1
                else:
                    # calculating argmax for R1 for each sample
                    R1[R1_temp > R1] = R1_temp[R1_temp > R1]

                if R2 is None:
                    R2 = R2_temp  # if 1st classifier, assign all vals to R1
                else:
                    # calculating argmax for R2 for each sample
                    R2[R2_temp > R2] = R2_temp[R2_temp > R2]

            # For each sample return minority class (1) if R1 >= R2, else return majority class (-1)
            return np.where(R1 >= R2, 1, -1)

        elif self.ensemble_rule == "min_dist":
            R1 = None
            R2 = None

            # for each classifier and subset i
            for clf, partition_samples, partition_classes in self.ensemble:
                # calculating P(i,1) and P(i,2)
                clf_probabilities = clf.predict_proba(samples)

                clf_avg_dist = []
                for sample in samples:
                    sample = np.asarray([sample])  # cdist requires sample to be a 2d array

                    # calculating D(i,1)
                    minority_samples = partition_samples[partition_classes == 1]
                    minority_distances = cdist(sample, minority_samples, metric="euclidean")[0]
                    average_minority_distance = minority_distances.mean()

                    # calculating D(i,2)
                    majority_samples = partition_samples[partition_classes == -1]
                    majority_distances = cdist(sample, majority_samples, metric="euclidean")[0]
                    average_majority_distance = majority_distances.mean()

                    # Recording D(i,j)
                    clf_avg_dist.append([average_majority_distance, average_minority_distance])

                clf_avg_dist = np.asarray(clf_avg_dist)
                # Pi1 / (Di1 + 1)
                R1_temp = clf_probabilities[:, 1] / (clf_avg_dist[:, 1] + 1)
                # Pi2 / (Di2 + 1)
                R2_temp = clf_probabilities[:, 0] / (clf_avg_dist[:, 0] + 1)
                del clf_avg_dist

                if R1 is None:
                    R1 = R1_temp  # if 1st classifier, assign all vals to R1
                else:
                    # calculating argmin for R1 for each sample
                    R1[R1_temp < R1] = R1_temp[R1_temp < R1]

                if R2 is None:
                    R2 = R2_temp  # if 1st classifier, assign all vals to R1
                else:
                    # calculating argmin for R2 for each sample
                    R2[R2_temp < R2] = R2_temp[R2_temp < R2]

            # For each sample return minority class (1) if R1 >= R2, else return majority class (-1)
            return np.where(R1 >= R2, 1, -1)

        elif self.ensemble_rule == "kitter_max":
            probabilities = []
            for clf, _, _ in self.ensemble:  # subset_samples and subset_classes not used in this ensemble, so dump to unused _ variable
                probabilities.append(clf.predict_proba(samples))
            probabilities = np.asarray(probabilities)

            transformed_probabilities = np.moveaxis(probabilities, 0, -1)
            best_maj_probability = np.max(transformed_probabilities[:, 0], 1)
            best_min_probability = np.max(transformed_probabilities[:, 1], 1)

            return np.where(best_min_probability >= best_maj_probability, 1, -1)


class DataBoostIM:
    def __init__(self, base_estimator, n_iterations=10):
        self.ensemble = []
        self.base_estimator = base_estimator
        self.n_iterations = n_iterations
        self.class_ids = None

    # X: List of training samples
    # Y: List of true sample classes {-1, 1}
    def fit(self, X, Y):
        # Reset ensemble object incase fit has already been called
        self.ensemble = []
        # Initialising variables
        T = range(self.n_iterations)
        m = len(X)
        D = np.full((self.n_iterations, m), 1. / m)
        # Initialising class_ids for generation stage
        self.class_ids = np.fromiter(set(Y), int)
        self.class_ids.sort()

        # Execute boosting rounds
        for t in T:
            # get sample weights for this iteration Dt
            Dt = D[t, :]
            # make copies of training set & weights, to be used for this current iteration only
            X_current = copy.deepcopy(X)
            Y_current = copy.deepcopy(Y)
            Dt_current = copy.deepcopy(Dt)

            # Perform DataBoost-IM synthesis if not first round
            if t > 0:
                # Identify seeds
                seeds = self.select_seeds(X_current, Y_current, Dt_current)

                # Generate synthetic samples
                syn_samples, syn_classes, syn_weights = \
                    self.generate_synthetic_samples_wrapper(X_current, Y_current, seeds)

                # merge synth data into original dataset and balance weights
                X_current, Y_current, Dt_current = \
                    self.merge_and_balance(X_current, Y_current, Dt_current, syn_samples, syn_classes, syn_weights)

            # get new copy of base classifier
            base_clf = copy.deepcopy(self.base_estimator)
            # train classifier with current training set and weights
            base_clf.fit(X_current, Y_current, Dt_current)

            # get classifier predictions for original training set
            htx = base_clf.predict(X)

            # calculate weighted error of ht
            error_t = np.sum(Dt[htx != Y])

            # if error is too high, abort training
            if error_t > 0.5:
                # if self.debugPrintGenerationStats:
                #     print("\t\t\t Error rate of base clf is too high (" + str(error_t) + ", aborting")
                break

            # calculate beta
            beta_t = (error_t + 1e-10) / (1 - error_t + 1e-10)

            # calculate error rate %
            err_rate = sum(htx != Y) / len(Y)
            # add base classifier, beta, error rate % to ensemble
            self.ensemble.append((base_clf, beta_t, err_rate))

            # If not already at the last iteration, increment sample weights for next iteration
            if t != (self.n_iterations - 1):
                # multiplying correctly predicted weights by beta
                Dt[htx == Y] = Dt[htx == Y] * beta_t
                # normalisation factor
                Zt = sum(Dt)
                # update sample weights
                D[t + 1, :] = Dt / Zt

    # X: List of training samples
    # Y: List of true sample classes {-1, 1}
    # Dt: List of sample weights
    def select_seeds(self, X, Y, Dt):
        # identify number of hard examples
        error_rate = self.ensemble[-1][2]
        Ns = round(len(X) * error_rate)

        # obtaining hard examples
        Es_indicies = np.flip(np.argsort(Dt))
        Es_weights = Dt[Es_indicies][:Ns]
        Es_classes = Y[Es_indicies][:Ns]

        # obtaining components needed to generate seeds
        Nmaj = len(Y[Y == -1])
        Nmin = len(Y[Y == 1])
        Nsmaj = len(Es_classes[Es_classes == -1])
        Nsmin = len(Es_classes[Es_classes == 1])
        Ml = min(round(Nmaj / Nmin), Nsmaj)
        Ms = min(round((Nmaj * Ml) / Nmin), Nsmin)

        # calculating the expected amount of synth data
        Nmaj_synthetic = round(Ml * Nmaj)  # number of majority class synthetic cases to be generated
        Nmin_synthetic = round(Ms * Nmin)  # number of minority class synthetic cases to be generated

        Esmaj_idx = np.flip(np.argsort(Es_weights * (Es_classes == -1)))
        Emaj = Esmaj_idx[:Ml]

        Esmin_idx = np.flip(np.argsort(Es_weights * (Es_classes == 1)))
        Emin = Esmin_idx[:Ms]

        # Return concatenated list of ordered seed weights (Emin & Emaj), along with their class labels
        return np.asarray([(Es_weights[w], 1) for w in Emin] + [(Es_weights[w], -1) for w in Emaj])

    # X: List of training samples
    # Y: List of true sample classes {-1, 1}
    # seeds: List of seeds (seed weight, class_id)
    def generate_synthetic_samples_wrapper(self, X, Y, seeds):
        synthetic_samples = []
        synthetic_weights = []
        synthetic_classes = []

        if len(seeds) > 0:
            # for each seed instance
            for seed_weight, class_id in seeds:
                # generate synthetic samples
                seed_synth_samples = self.generate_synthetic_sample_set_for_seed(X, Y, class_id)
                assert len(seed_synth_samples) == len(Y[Y == class_id]), "Wrong no of synth samples"
                n_samples_generated = len(seed_synth_samples)
                # calculate synthetic sample weights
                seed_synth_weights = np.full(n_samples_generated, seed_weight / n_samples_generated)

                # append new samples to synthetic sample set
                synthetic_samples.append(seed_synth_samples)
                synthetic_weights.append(seed_synth_weights)
                synthetic_classes.append(np.full(n_samples_generated, class_id))

            # Flatten lists into np array
            synthetic_samples = np.concatenate(synthetic_samples)
            synthetic_classes = np.concatenate(synthetic_classes)
            synthetic_weights = np.concatenate(synthetic_weights)

        return synthetic_samples, synthetic_classes, synthetic_weights

    # X: List of training samples
    # Y: List of true sample classes {-1, 1}
    # class_id: ClassID to generate samples from
    def generate_synthetic_sample_set_for_seed(self, X, Y, class_id):
        synth_data = []

        # Loop through attribute values, rather than samples
        for dataset_attr_vals in np.transpose(X):
            # Get number of unique values contained by this attribute value
            unique_attr_vals = len(set(dataset_attr_vals))

            # binary data or homogenous/singular value data item
            if unique_attr_vals == 2 or unique_attr_vals == 1:
                class_attr_vals = dataset_attr_vals[Y == class_id]
                np.random.shuffle(class_attr_vals)
                synth_data.append(class_attr_vals)
            elif unique_attr_vals > 2:
                # assume continuous data, there's no nominal data in supplied datasets
                class_attr_vals = dataset_attr_vals[Y == class_id]
                attr_mean = np.mean(class_attr_vals)
                attr_std = np.std(class_attr_vals)
                attr_low = np.amin(class_attr_vals)
                attr_high = np.amax(class_attr_vals)

                # generate new synthetic data using gaussian distribution
                synth_attr_vals = self.generate_gaussian_distribution_within_range(
                    attr_mean, attr_std, attr_low, attr_high, len(class_attr_vals))
                np.random.shuffle(synth_attr_vals)
                synth_data.append(synth_attr_vals)
            else:
                raise ValueError("Unable to handle attribute during synthetic data generation")

        return np.transpose(np.asarray(synth_data))

    # mean: Distribution mean
    # std: Standard deviation of generated distribution
    # min_val: Minimum value allowed for generated distribution
    # max_val: Maximum value allowed for generated distribution
    # size: Size of generated distribution
    @staticmethod
    def generate_gaussian_distribution_within_range(mean, std, min_val, max_val, size):
        # Create gaussian distribution using np.random
        dist = np.random.normal(loc=mean, scale=std, size=size)

        # np.random creates gaussian distributions without a range limit, so repeatedly remove
        # samples that are outside of the specified range and generate new values to reach the
        # number of samples specified by "size".
        while len(dist[dist < min_val]) > 0 or len(dist[dist > max_val]) > 0 or len(dist) != size:
            if len(dist) != size:
                raise ValueError("Length of distribution did not match specified size during loop. Aborting to prevent infinite loop")

            # trim vals outside range
            dist = np.delete(dist, (dist < min_val))
            dist = np.delete(dist, (dist > max_val))

            # generate new vals to replace missing
            n_missing = size - len(dist)
            filler_dist = np.random.normal(loc=mean, scale=std, size=n_missing)
            dist = np.append(dist, filler_dist)

        return dist

    # X: List of current training samples
    # Y: List of current true sample classes {-1, 1}
    # Dt: List of current sample weights
    # syn_x: List of synthetic training samples to merge
    # syn_y: List of synthetic sample classes to merge
    # syn_dt: List of synthetic sample weights to merge
    @staticmethod
    def merge_and_balance(X, Y, Dt, syn_x, syn_y, syn_dt):
        # Merge synthetic data into original dataset, then balance class weights
        if len(syn_x) > 0:
            X = np.concatenate((X, syn_x), axis=0)
            Y = np.concatenate((Y, syn_y), axis=0)
            Dt = np.concatenate((Dt, syn_dt), axis=0)

            # Balance current sample weight distribution following dataset merge
            Wmin = sum(Dt[Y == 1])
            Wmaj = sum(Dt[Y == -1])
            if Wmaj > Wmin:
                Dt[Y == 1] = Dt[Y == 1] * (Wmaj / Wmin)
            else:
                Dt[Y == -1] = Dt[Y == -1] * (Wmin / Wmaj)

            # normalise distribution
            Dt = Dt / sum(Dt)

        # return merged and balanced dataset, or original dataset if no synth samples were passed
        return X, Y, Dt

    # X: List of samples to predict
    def predict(self, X):
        if len(self.ensemble) > 0:
            # initialising vars
            class_ids = copy.deepcopy(self.class_ids)
            ensemble_predictions = []
            log_inv_beta = []

            # gathering sample predictions from base classifiers
            for (ht, beta_t, _) in self.ensemble:
                htx = ht.predict(X)
                ensemble_predictions.append(htx)
                log_inv_beta.append(np.log(1. / beta_t))
            ensemble_predictions = np.asarray(ensemble_predictions)
            log_inv_beta = np.asarray(log_inv_beta)

            # gathering classifier predictions by sample, by class
            ensemble_predictions_by_class = []
            for y in class_ids:
                ensemble_predictions_of_class = np.where(ensemble_predictions == y, True, False)
                ensemble_predictions_by_class.append(ensemble_predictions_of_class)
            ensemble_predictions_by_class = np.asarray(ensemble_predictions_by_class)

            # multiplying each sample by the appropriate log(1/beta)
            ensemble_predictions_by_class_beta = ensemble_predictions_by_class * log_inv_beta[None, :, None]
            # sum of log(1/B) over each classifier for each class
            class_vals = np.sum(ensemble_predictions_by_class_beta, axis=1)

            # argmax for each class by sample
            best_classidx_by_sample = np.argmax(class_vals, 0)

            # retrieving predicted class labels
            Hx = class_ids[best_classidx_by_sample]

            return Hx
        else:
            print("\t\t No base classifiers found in DataBoost-IM ensemble object. Training must have failed.")
            return None


class CostSensitiveAdaBoost:
    def __init__(self, base_estimator, adac_type, n_iterations=10):
        self.ensemble = []
        self.n_iterations = n_iterations
        self.base_estimator = base_estimator
        self.adac_type = adac_type

    # X: List of training samples
    # Y: List of true sample classes {-1, 1}
    # C: List of costs for each sample in range [0,1]
    def fit(self, X, Y, C):
        # Reset ensemble object incase fit has already been called
        self.ensemble = []

        # Initialising variables
        T = range(self.n_iterations)
        m = len(X)
        D = np.full((self.n_iterations, m), 1. / m)
        # if self.adac_type == "adacost":
        #     D = np.full((self.n_iterations, len_x), C / sum(C))
        # else:
        #     D = np.full((self.n_iterations, len_x), 1. / len_x)

        # Iterating each base classifier
        for t in T:
            # get sample weights for this iteration
            Dt = D[t, :]
            # get new copy of base classifier
            base_clf = copy.deepcopy(self.base_estimator)
            # train classifier with weights
            base_clf.fit(X, Y, Dt)

            # get classifier predictions
            htx = base_clf.predict(X)

            # getting components of weight update parameter
            hit = htx == Y
            miss = htx != Y
            sum_hit_costed_weights = np.sum(Dt[hit] * C[hit])
            sum_miss_costed_weights = np.sum(Dt[miss] * C[miss])

            # Calculating factor of weight update parameter for each version of AdaCx
            if self.adac_type == "adac1":
                if sum_hit_costed_weights <= sum_miss_costed_weights:
                    print("\t\t Error too high, aborting...")
                    break
                factor_numerator = 1 + sum_hit_costed_weights - sum_miss_costed_weights
                factor_denomenator = 1 - sum_hit_costed_weights + sum_miss_costed_weights
                factor = factor_numerator / factor_denomenator

            if self.adac_type == "adac2":
                if sum_hit_costed_weights <= sum_miss_costed_weights:
                    print("\t\t Error too high, aborting...")
                    break
                factor = sum_hit_costed_weights / sum_miss_costed_weights

            if self.adac_type == "adac3":
                sum_hit_c_sq_weights = np.sum(Dt[hit] * C[hit] * C[hit])
                sum_miss_c_sq_weights = np.sum(Dt[miss] * C[miss] * C[miss])

                if sum_hit_c_sq_weights <= sum_miss_c_sq_weights:
                    print("\t\t Error too high, aborting...")
                    break

                sum_of_costed_weights = np.sum(Dt * C)
                factor_numerator = (sum_of_costed_weights + sum_hit_c_sq_weights - sum_miss_c_sq_weights)
                factor_denomenator = (sum_of_costed_weights - sum_hit_c_sq_weights + sum_miss_c_sq_weights)
                factor = factor_numerator / factor_denomenator

            if self.adac_type == "adacost":
                beta = self.beta(htx, Y, C)
                r = sum(Dt * Y * htx * beta)
                factor = (1. + r) / (1. - r)

            # Calculating weight update parameter
            alpha_t = 0.5 * np.log(factor)

            # add base classifier and weight update parameter to ensemble
            self.ensemble.append((base_clf, alpha_t, sum_miss_costed_weights))

            # If not already at the last iteration, increment sample weights for next iteration
            if t != (self.n_iterations - 1):
                # Updating sample weights
                if self.adac_type == "adac1":
                    numerator = Dt * np.exp(-alpha_t * Y * C * htx)
                elif self.adac_type == "adac2":
                    numerator = C * Dt * np.exp(-alpha_t * Y * htx)
                elif self.adac_type == "adac3":
                    numerator = C * Dt * np.exp(-alpha_t * Y * C * htx)
                elif self.adac_type == "adacost":
                    # Reverse polarity of predictions if alpha_t < 0
                    if alpha_t < 0:
                        alpha_t = -alpha_t
                        htx = -htx
                    numerator = Dt * np.exp(-alpha_t * Y * htx * self.beta(htx, Y, C))

                Zt = np.sum(numerator)
                D[t + 1, :] = numerator / Zt

    # X: List of samples to predict
    def predict(self, X):
        if len(self.ensemble) > 0:
            # gathering weighted predictions for every sample from all base classifiers
            raw_ensemble_predictions = []
            for (ht, alpha_t, error) in self.ensemble:
                htx = ht.predict(X)
                raw_ensemble_predictions.append(htx.astype(np.float_) * alpha_t)

            # get majority vote prediction for samples, int {-1, 1}
            Hx = np.sign(np.sum(raw_ensemble_predictions, axis=0))
            return Hx
        else:
            print("\t\t No base classifiers found in " + self.adac_type + "ensemble object. Training must have failed.")
            return None

    # htx: List of predicted sample classes
    # Y: List of true sample classes
    # C: List of sample costs
    @staticmethod
    def beta(htx, Y, C):
        sign = htx * Y
        beta = ((-1 * sign) * 0.5 * C) + 0.5
        return beta


class Adaboost:
    def __init__(self, base_estimator, n_iterations=10):
        self.ensemble = []
        self.n_iterations = n_iterations
        self.base_estimator = base_estimator

    # X: List of training samples
    # Y: List of true sample classes {-1, 1}
    def fit(self, X, Y):
        # Reset ensemble object incase fit has already been called
        self.ensemble = []

        # Initialising variables
        T = range(self.n_iterations)
        m = len(X)
        D = np.full((self.n_iterations, m), 1. / m)

        # Execute boosting rounds
        for t in T:
            # get sample weights for this iteration Dt
            Dt = D[t, :]
            # get new copy of base classifier
            ht = copy.deepcopy(self.base_estimator)
            # train classifier with weights
            ht.fit(X, Y, Dt)

            # get classifier predictions
            htx = ht.predict(X)

            # getting components of weight update parameter
            hit = htx == Y
            miss = htx != Y
            sum_hit_weights = np.sum(Dt[hit])
            sum_miss_weights = np.sum(Dt[miss])

            if sum_hit_weights <= sum_miss_weights:
                print("\t\t\t error too high, aborting")
                break

            # calculating weight update parameter
            factor = (sum_hit_weights + 1e-10) / (sum_miss_weights + 1e-10)
            alpha_t = 0.5 * np.log(factor)

            # add base classifier, weight update parameter, and error to ensemble
            self.ensemble.append((ht, alpha_t, sum_miss_weights))

            # If not already at the last iteration, increment sample weights for next iteration
            if t != (self.n_iterations - 1):
                numerator = Dt * np.exp(-alpha_t * Y * htx)
                # normalisation factor
                Zt = np.sum(numerator)
                # update sample weights
                D[t + 1, :] = numerator / Zt

    # X: List of samples to predict
    def predict(self, X):
        raw_ensemble_predictions = []

        # gathering weighted predictions for every sample from all base classifiers
        for (ht, alpha_t, error) in self.ensemble:
            htx = ht.predict(X)
            raw_ensemble_predictions.append(htx.astype(np.float_) * alpha_t)

        # get majority vote prediction for samples, int {-1, 1}
        Hx = np.sign(np.sum(raw_ensemble_predictions, axis=0))
        return Hx
