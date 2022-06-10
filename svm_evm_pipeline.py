# Training and testing SVM and EVM using generated network features

import time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import svm, datasets
from sklearn.metrics import top_k_accuracy_score
from sklearn import linear_model

# EVM
from vast.opensetAlgos.EVM import EVM_Training, EVM_Inference

# Derek's wrapper
from vast.opensetAlgos.extreme_value_machine import ExtremeValueMachine


####################################################
# Parameters
####################################################
kernel = "linear"
degree = 3
c = 1
cov_thresh = [0.1, 0.2, 0.3, 0.4, 0.5,
              0.6, 0.7, 0.8, 0.9]
tail_size = [10, 100, 1000, 10000]
percent = 50
pca_ratio = 0.99

device = "cuda:0"

run_svm = True
run_evm = False

debug = False
start = time.time()

####################################################
# Data paths
####################################################
# Temporary debug data (from previous video data)
if debug:
    train_known_known_feature_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                                     "evm/data/ucf101_ta2/ucf_101_train_known_feature.npy"
    train_known_known_label_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                                   "evm/data/ucf101_ta2/ucf_101_train_known_label.npy"

    test_known_known_feature_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                                    "evm/data/ucf101_ta2/ucf_101_test_known_valid_feature.npy"
    test_known_known_label_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                                  "evm/data/ucf101_ta2/ucf_101_test_known_valid_label.npy"

    test_known_known_feature = np.load(test_known_known_feature_path)
    test_known_known_label = np.load(test_known_known_label_path)

    test_unknown_unknown_feature_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                                        "evm/data/ucf101_ta2/ucf_101_test_unknown_valid_feature.npy"

    save_result_dir = "/afs/crc.nd.edu/user/j/jhuang24/vast_debug_files"

    valid_known_known_probs_path = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/models/msd_net/" \
                                   "2022-02-13/known_only_cross_entropy/seed_0/features/valid_known_known_epoch_147_probs.npy"

# Data path (our real features)
else:
    # TODO: cross entropy
    model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_0", 147
    # model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_1", 181
    # model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_2", 195
    # model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_3", 142
    # model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_4", 120

    feature_base_scratch = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/msd_net"
    feature_base_home = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/models/msd_net"

    # Train and test feature path (with epoch index)
    train_known_known_feature_path = feature_base_scratch + "/" + model_dir + "/test_results/train_known_known_epoch_" + str(epoch) + "_features_reduced.npy"
    train_known_known_label_path = feature_base_home + "/" + model_dir + "/features/train_known_known_epoch_" + str(epoch) + "_labels.npy"
    train_known_known_feature = np.load(train_known_known_feature_path)
    train_known_known_label = np.load(train_known_known_label_path)
    print("train feature loaded")

    valid_known_known_feature_path = feature_base_scratch + "/" + model_dir + "/test_results/valid_known_known_epoch_" + str(epoch) + "_features_reduced.npy"
    valid_known_known_label_path = feature_base_home + "/" + model_dir + "/features/valid_known_known_epoch_" + str(epoch) + "_labels.npy"
    valid_known_known_probs_path = feature_base_home + "/" + model_dir + "/features/valid_known_known_epoch_" + str(epoch) + "_probs.npy"

    valid_known_known_feature = np.load(valid_known_known_feature_path)
    valid_known_known_label = np.load(valid_known_known_label_path)
    t1 = time.time()
    print("check 1: ", (t1 - start))
    print("valid feature and probs loaded")

    test_known_known_feature_path_p0 = feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_features_p0_reduced.npy"
    test_known_known_label_path_p0 = feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_part_0_labels.npy"
    test_known_known_feat_p0 = np.load(test_known_known_feature_path_p0)
    test_known_known_labels_p0 = np.load(test_known_known_label_path_p0)
    t2 = time.time()
    print("check 2: ", (t2 - t1))
    print("test known p0 loaded")

    test_known_known_feature_path_p1 = feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_features_p1_reduced.npy"
    test_known_known_label_path_p1 = feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_part_1_labels.npy"
    test_known_known_feat_p1 = np.load(test_known_known_feature_path_p1)
    test_known_known_labels_p1 = np.load(test_known_known_label_path_p1)
    t3 = time.time()
    print("check 3: ", (t3 - t2))
    print("test known p1 loaded")

    test_known_known_feature_path_p2 = feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_features_p2_reduced.npy"
    test_known_known_label_path_p2 = feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_part_2_labels.npy"
    test_known_known_feat_p2 = np.load(test_known_known_feature_path_p2)
    test_known_known_labels_p2 = np.load(test_known_known_label_path_p2)
    t4 = time.time()
    print("check 4: ", (t4 - t3))
    print("test known p2 loaded")

    test_known_known_feature_path_p3 = feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_features_p3_reduced.npy"
    test_known_known_label_path_p3 = feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_part_3_labels.npy"
    test_known_known_feat_p3 = np.load(test_known_known_feature_path_p3)
    test_known_known_labels_p3 = np.load(test_known_known_label_path_p3)
    t5 = time.time()
    print("check 5: ", (t5 - t4))
    print("test known p3 loaded")

    test_unknown_unknown_feature_path = feature_base_scratch + "/" + model_dir + "/test_results/test_unknown_unknown_epoch_" + str(epoch) +"_features_reduced.npy"
    test_unknown_unknown_feature = np.load(test_unknown_unknown_feature_path)
    print("test unknown loaded")

    # Paths to save EVM model, EVM probs and results
    save_result_dir = feature_base_scratch + "/" + model_dir


#####################################################
# Load the features and labels and check their shape
#####################################################
train_known_known_feature = np.load(train_known_known_feature_path)
train_known_known_label = np.load(train_known_known_label_path)

if debug:
    train_known_known_feature = np.load(train_known_known_feature_path)
    train_known_known_label = np.load(train_known_known_label_path)
    test_unknown_unknown_feature = np.load(test_unknown_unknown_feature_path)

    train_known_known_feature = np.reshape(train_known_known_feature,
                                           (train_known_known_feature.shape[0],
                                            train_known_known_feature.shape[1]*train_known_known_feature.shape[2]))

    test_known_known_feature = np.reshape(test_known_known_feature,
                                         (test_known_known_feature.shape[0],
                                          test_known_known_feature.shape[1]*test_known_known_feature.shape[2]))

    test_unknown_unknown_feature = np.reshape(test_unknown_unknown_feature,
                                             (test_unknown_unknown_feature.shape[0],
                                              test_unknown_unknown_feature.shape[1] * test_unknown_unknown_feature.shape[2]))

    # For testing, just get 100 samples
    train_known_known_feature = train_known_known_feature[:100]
    valid_known_known_feature = train_known_known_feature
    test_known_known_feature = test_known_known_feature[:30]
    test_unknown_unknown_feature = test_unknown_unknown_feature[:100]

    train_known_known_label = train_known_known_label[:100]
    valid_known_known_label = train_known_known_label
    test_known_known_label = test_known_known_label[:30]

else:
    pass

print("Train known known:", train_known_known_feature.shape, train_known_known_label.shape)
print("Valid known known", valid_known_known_feature.shape, valid_known_known_label.shape)
print("Test unknown unknown:", test_unknown_unknown_feature.shape)


####################################################
# Getting threshold
####################################################
def get_thresholds(npy_file_path,
                   percentile):
    """
    Get the probability thresholds for 5 exits respectively.

    :param npy_file_path:
    :return:
    """
    prob_clf_0 = []
    prob_clf_1 = []
    prob_clf_2 = []
    prob_clf_3 = []
    prob_clf_4 = []


    # Load npy file and check the shape
    probs = np.load(npy_file_path)
    # print(probs.shape) # Shape [nb_samples, nb_clfs, nb_classes]

    # Process each sample
    for i in range(probs.shape[0]):
        one_sample_probs = probs[i, :, :]
        one_sample_probs = np.reshape(one_sample_probs,(probs.shape[1],
                                                        probs.shape[2]))

        # Check each classifier in each sample
        for j in range(one_sample_probs.shape[0]):
            one_clf_probs = one_sample_probs[j, :]

            # Find the max prob
            max_prob = np.max(one_clf_probs)

            if j == 0:
                prob_clf_0.append(max_prob)
            elif j == 1:
                prob_clf_1.append(max_prob)
            elif j == 2:
                prob_clf_2.append(max_prob)
            elif j == 3:
                prob_clf_3.append(max_prob)
            elif j == 4:
                prob_clf_4.append(max_prob)

    thresh_0 = np.percentile(np.asarray(prob_clf_0), percentile)
    thresh_1 = np.percentile(np.asarray(prob_clf_1), percentile)
    thresh_2 = np.percentile(np.asarray(prob_clf_2), percentile)
    thresh_3 = np.percentile(np.asarray(prob_clf_3), percentile)
    thresh_4 = np.percentile(np.asarray(prob_clf_4), percentile)

    return [thresh_0, thresh_1, thresh_2, thresh_3, thresh_4]




def get_unknown_acc(probs,
                    threshold):
    """

    :param probs: nb_sample x nb_classes
    :param threshold:
    :return:
    """
    nb_correct = 0

    for i in range(probs.shape[0]):
        max_prob = np.max(probs[i])

        if max_prob < threshold:
            nb_correct += 1

    acc = float(nb_correct)/float(probs.shape[0])
    print("Test unknown unknown acc: ", acc)

    return acc




# TODO: this function TBD - do we need it or not?
def get_known_acc(labels,
                  probs):
    nb_correct_top_1 = 0
    nb_correct_top_3 = 0
    nb_correct_top_5 = 0

    for i in range(len(labels)):
        target_label = labels[i]
        one_prob = probs[i]

        top_1 = np.argmax(one_prob)
        top_3 = np.argpartition(one_prob, -3)[-3:]
        top_5 = np.argpartition(one_prob, -5)[-5:]

        if top_1 == target_label:
            nb_correct_top_1 += 1
            nb_correct_top_3 += 1
            nb_correct_top_5 += 1

        elif target_label in top_3:
            nb_correct_top_3 += 1
            nb_correct_top_5 += 1

        elif target_label in top_5:
            nb_correct_top_5 += 1

        else:
            continue

    acc_top_1 = float(nb_correct_top_1) / (float(len(labels)))
    acc_top_3 = float(nb_correct_top_3) / (float(len(labels)))
    acc_top_5 = float(nb_correct_top_5) / (float(len(labels)))

    return acc_top_1, acc_top_3, acc_top_5




def calculate_mcc(true_pos,
                  true_neg,
                  false_pos,
                  false_neg):
    """

    :param true_pos:
    :param true_neg:
    :param false_pos:
    :param false_negtive:
    :return:
    """

    return (true_neg*true_pos-false_pos*false_neg)/np.sqrt((true_pos+false_pos)*(true_pos+false_neg)*
                                                           (true_neg+false_pos)*(true_neg+false_neg))




def get_binary_stats(test_known_prob,
                     test_unknown_prob,
                     threshold):
    """

    :param test_known_feature:
    :param test_known_label:
    :param test_unknown_feature:
    :param nb_test_parts:
    :return:
    """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    if test_known_prob is not None:
        for one_prob_set in test_known_prob:
            one_prob = np.max(one_prob_set)

            if one_prob > threshold:
                true_positive += 1
            else:
                false_negative += 1

    if test_unknown_prob is not None:
        for one_prob_set in test_unknown_prob:
            one_prob = np.max(one_prob_set)

            if one_prob > threshold:
                false_positive += 1
            else:
                true_negative += 1

        print([true_positive, true_negative, false_positive, false_negative])

    return true_positive, true_negative, false_positive, false_negative




def train_test_svm(train_known_feature,
                   train_known_labels,
                   valid_known_feature,
                   valid_known_labels,
                   test_known_feature,
                   test_known_labels,
                   test_unknown_feature,
                   nb_test_parts,
                   debug,
                   novelty_thresh):
    """

    :param train_known_feature:
    :param train_known_labels:
    :param valid_known_feature:
    :param valid_known_label:
    :param test_known_feature:
    :param test_known_labels:
    :param test_unknown_feature:
    :param debug:
    :return:
    """
    print("Training SVM...")
    svm_model = linear_model.SGDClassifier(loss="hinge",
                              penalty="l2",
                              max_iter=1000,
                              class_weight='balanced').fit(train_known_feature, train_known_labels)

    # Test SVM
    print("Testing SVM...")

    if debug:
        pred_valid = svm_model.predict(valid_known_feature)
        pred_known_known = svm_model.predict(test_known_feature)
        unknown_unknown_prob = svm_model.predict_proba(test_unknown_feature)

        valid_acc_known_top_1 = top_k_accuracy_score(valid_known_labels, pred_valid, k=1)
        test_acc_known_top_1 = top_k_accuracy_score(test_known_labels, pred_known_known, k=1)

        return valid_acc_known_top_1, 0.000, 0.000, \
               test_acc_known_top_1, 0.000, 0.000, \
               unknown_unknown_prob

    else:
        #################################################################
        # valid results
        #################################################################
        pred_valid = svm_model.decision_function(valid_known_feature)

        valid_acc_known_top_1, \
        valid_acc_known_top_3,\
        valid_acc_known_top_5 = get_known_acc(labels=valid_known_labels,
                                              probs=pred_valid)

        #################################################################
        # Test known
        #################################################################
        test_top_1_acc = []
        test_top_3_acc = []
        test_top_5_acc = []

        true_positive = []
        true_negative = []
        false_positive = []
        false_negative = []

        for i in range(nb_test_parts):
            one_label = test_known_labels[i]
            one_feature = test_known_feature[i]
            pred_known_known = svm_model.decision_function(one_feature)

            # calculate mutil-class accuracy
            test_acc_known_top_1, \
            test_acc_known_top_3, \
            test_acc_known_top_5 = get_known_acc(labels=one_label,
                                                  probs=pred_known_known)

            test_top_1_acc.append(test_acc_known_top_1)
            test_top_3_acc.append(test_acc_known_top_3)
            test_top_5_acc.append(test_acc_known_top_5)

            # Calculate binary
            tp, tn, fp, fn = get_binary_stats(test_known_prob=pred_known_known,
                                                test_unknown_prob=None,
                                                threshold=novelty_thresh)
            true_positive.append(tp)
            true_negative.append(tn)
            false_positive.append(fp)
            false_negative.append(fn)

        test_acc_known_top_1 = sum(test_top_1_acc)/len(test_top_1_acc)
        test_acc_known_top_3 = sum(test_top_3_acc)/len(test_top_3_acc)
        test_acc_known_top_5 = sum(test_top_5_acc)/len(test_top_5_acc)

        #################################################################
        # Test unknown
        #################################################################
        unknown_unknown_prob = svm_model.decision_function(test_unknown_feature)
        tp, tn, fp, fn = get_binary_stats(test_known_prob=None,
                                          test_unknown_prob=unknown_unknown_prob,
                                          threshold=novelty_thresh)

        true_positive.append(tp)
        true_negative.append(tn)
        false_positive.append(fp)
        false_negative.append(fn)

        #################################################################
        # Output all results
        #################################################################
        true_positive = sum(true_positive)
        true_negative = sum(true_negative)
        false_positive = sum(false_positive)
        false_negative = sum(false_negative)

        print("True positive: ", true_positive)
        print("True negative: ", true_negative)
        print("False postive: ", false_positive)
        print("False negative: ", false_negative)

        mcc = calculate_mcc(true_pos=float(true_positive),
                            true_neg=float(true_negative),
                            false_pos=float(false_positive),
                            false_neg=float(false_negative))

        precision = float(true_positive) / float(true_positive + false_positive)
        recall = float(true_positive) / float(true_positive + false_negative)
        f1 = (2 * precision * recall) / (precision + recall)
        unknown_acc = float(true_negative) / float(true_negative + false_positive)

        print("Test unknown accuracy: ", unknown_acc)
        print("F-1 score: ", f1)
        print("MCC score: ", mcc)
        print("Test known acc top-1: ", test_acc_known_top_1)
        print("Test known acc top-3: ", test_acc_known_top_3)
        print("Test known acc top-5: ", test_acc_known_top_5)





def train_test_evm(train_known_feature,
                   train_known_labels,
                   valid_known_feature,
                   valid_known_labels,
                   test_known_feature,
                   test_known_labels,
                   test_unknown_feature,
                   tail_size,
                   debug,
                   cover_threshold,
                   nb_test_parts=4):
    """

    :param train_known_feature:
    :param train_known_labels:
    :param test_known_feature:
    :param test_unknown_feature:
    :return:
    """
    """
    evm.predict: nb_sample x (nb_class+1)
    evm.known_probs: nb_sample x nb_class
    """

    labels = list(np.unique(train_known_labels))

    # Define EVM
    print("Creating EVM... Current tail size is %d" % tail_size)
    evm = ExtremeValueMachine(tail_size=tail_size,
            cover_threshold=cover_threshold,
            distance_multiplier=1.0,
            labels=labels,
            distance_metric="cosine",
            chunk_size=200,
            device=device,
            tail_size_is_ratio=False)

    # Train(fit) EVM
    print("Training EVM")
    evm.fit(train_known_feature, train_known_labels)

    # Save EVM model, so no need to retrain
    evm_model_save_path = save_result_dir + "/evm_model_tail_size_" + str(tail_size) + ".pkl"
    print("Saving EVM model")
    evm.save(evm_model_save_path)

    # Convert data type and Test EVM
    print("Testing EVM")

    valid_known_known_feature = torch.from_numpy(valid_known_feature).float()
    valid_known_known_known_probs = evm.known_probs(valid_known_known_feature).cpu().detach().numpy()

    test_unknown_unknown_feature = torch.from_numpy(test_unknown_feature).float()
    unknown_unknown_known_probs = evm.known_probs(test_unknown_unknown_feature).cpu().detach().numpy()

    # Known accuracy
    if debug:
        test_known_known_feature = torch.from_numpy(test_known_feature).float()
        test_known_known_known_probs = evm.known_probs(test_known_known_feature).cpu().detach().numpy()

        valid_pred_labels = np.argmax(valid_known_known_known_probs, axis=1)
        test_pred_labels = np.argmax(test_known_known_known_probs, axis=1)

        valid_acc_known_top_1 = top_k_accuracy_score(valid_known_labels, valid_pred_labels, k=1)
        test_acc_known_top_1 = top_k_accuracy_score(test_known_labels, test_pred_labels, k=1)

        return valid_acc_known_top_1, 0.000, 0.000,\
               test_acc_known_top_1, 0.000, 0.000, \
               valid_known_known_known_probs, test_known_known_known_probs, unknown_unknown_known_probs

    else:
        valid_acc_known_top_1, \
        valid_acc_known_top_3, \
        valid_acc_known_top_5 = get_known_acc(labels=valid_known_labels,
                                              probs=valid_known_known_known_probs)

        test_top_1_acc = []
        test_top_3_acc = []
        test_top_5_acc = []

        for i in range(nb_test_parts):
            one_label = test_known_labels[i]
            one_feature = test_known_feature[i]

            one_feature = torch.from_numpy(one_feature).float()
            pred_known_known = evm.known_probs(one_feature).cpu().detach().numpy()

            test_acc_known_top_1, \
            test_acc_known_top_3, \
            test_acc_known_top_5 = get_known_acc(labels=one_label,
                                                 probs=pred_known_known)

            test_top_1_acc.append(test_acc_known_top_1)
            test_top_3_acc.append(test_acc_known_top_3)
            test_top_5_acc.append(test_acc_known_top_5)

        test_acc_known_top_1 = sum(test_top_1_acc) / len(test_top_1_acc)
        test_acc_known_top_3 = sum(test_top_3_acc) / len(test_top_3_acc)
        test_acc_known_top_5 = sum(test_top_5_acc) / len(test_top_5_acc)

        return valid_acc_known_top_1, valid_acc_known_top_3,  valid_acc_known_top_5, \
               test_acc_known_top_1, test_acc_known_top_3, test_acc_known_top_5, \
               unknown_unknown_known_probs




if __name__ == '__main__':
    # Get threshold for novelty
    novelty_thresh = get_thresholds(npy_file_path=valid_known_known_probs_path,
                                    percentile=percent)

    # Option for svm
    if run_svm:
        train_test_svm(train_known_feature=train_known_known_feature,
                      train_known_labels=train_known_known_label,
                      valid_known_feature=valid_known_known_feature,
                      valid_known_labels=valid_known_known_label,
                       test_known_feature=[test_known_known_feat_p0,
                                           test_known_known_feat_p1,
                                           test_known_known_feat_p2,
                                           test_known_known_feat_p3],
                       test_known_labels=[test_known_known_labels_p0,
                                          test_known_known_labels_p1,
                                          test_known_known_labels_p2,
                                          test_known_known_labels_p3],
                       test_unknown_feature=test_unknown_unknown_feature,
                       nb_test_parts=4,
                       debug=debug,
                       novelty_thresh=novelty_thresh[-1])

    # Option for evm
    if run_evm:
        for one_cover_threshold in cov_thresh:
            for one_tail in tail_size:
                print("*" * 50)
                print("current cover threshold: ", one_cover_threshold)
                print("current tail size: ", one_tail)

                evm_valid_top_1, evm_valid_top_3, \
                evm_valid_top_5, evm_test_top_1, \
                evm_test_top_3, evm_test_top_5, \
                evm_unknown_unknown_known_probs = train_test_evm(train_known_feature=train_known_known_feature,
                                                                train_known_labels=train_known_known_label,
                                                                valid_known_feature=valid_known_known_feature,
                                                                valid_known_labels=valid_known_known_label,
                                                                   test_known_feature=[test_known_known_feat_p0,
                                                                                        test_known_known_feat_p1,
                                                                                        test_known_known_feat_p2,
                                                                                        test_known_known_feat_p3],
                                                                   test_known_labels=[test_known_known_labels_p0,
                                                                                        test_known_known_labels_p1,
                                                                                        test_known_known_labels_p2,
                                                                                        test_known_known_labels_p3],
                                                                   test_unknown_feature=test_unknown_unknown_feature,
                                                                   tail_size=one_tail,
                                                                   cover_threshold=one_cover_threshold,
                                                                   debug=debug)

                # EVM post-process for unknown
                evm_acc_unknown_unknown = get_unknown_acc(probs=evm_unknown_unknown_known_probs,
                                                          threshold=novelty_thresh[-1])



