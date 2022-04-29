# Training and testing SVM and EVM using generated network features


from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
# import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
# import matplotlib.pyplot as plt
import h5py
from sklearn.decomposition import IncrementalPCA

from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sys
import os

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
tail_size = [10, 100, 1000, 10000]

device = "cuda:0"

run_svm = True
run_evm = True

debug = True

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

    test_unknown_unknown_feature_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                                        "evm/data/ucf101_ta2/ucf_101_test_unknown_valid_feature.npy"

    evm_model_save_path = "/afs/crc.nd.edu/user/j/jhuang24/debug_evm_model.pkl"

# Data path (our real features)
else:
    # TODO: cross entropy
    model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_0", 147
    # model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_1", 181
    # model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_2", 195
    # model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_3", 142
    # model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_4", 120

    # TODO: Cross-entropy + sam
    # model_dir, epoch = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_0", 175
    # model_dir, epoch = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_1", 105
    # model_dir, epoch = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_2", 159
    # model_dir, epoch = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_3", 103
    # model_dir, epoch = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_4", 193

    # TODO: All 3 losses
    # model_dir, epoch = "2022-03-30/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_0", 156
    # model_dir, epoch = "2022-03-30/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_1", 194
    # model_dir, epoch = "2022-03-30/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_2", 192
    # model_dir, epoch = "2022-03-25/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_3", 141
    # model_dir, epoch = "2022-03-25/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_4", 160

    # TODO: CE + pp
    # model_dir, epoch = "2022-03-25/cross_entropy_1.0_exit_1.0_unknown_ratio_1.0/seed_0", 173
    # model_dir, epoch = "2022-03-30/cross_entropy_1.0_exit_1.0_unknown_ratio_1.0/seed_1", 130
    # model_dir, epoch = "2022-03-30/cross_entropy_1.0_exit_1.0_unknown_ratio_1.0/seed_2", 166
    # model_dir, epoch = "2022-03-30/cross_entropy_1.0_exit_1.0_unknown_ratio_1.0/seed_3", 128
    # model_dir, epoch = "2022-03-30/cross_entropy_1.0_exit_1.0_unknown_ratio_1.0/seed_4", 110

    known_feature_base = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/models/msd_net"
    unknown_feature_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/msd_net"

    known_feature_dir = os.path.join(known_feature_base, model_dir)
    unknown_feature_dir = os.path.join(unknown_feature_base, model_dir)

    # Train and test feature path (with epoch index)
    train_known_known_feature_path = known_feature_dir + "/features/train_known_known_" + str(epoch) + "_features.npy"
    train_known_known_label_path = known_feature_dir + "/features/train_known_known_" + str(epoch) + "_labels.npy"

    test_known_known_feature_path = known_feature_dir + "/features/test_known_known_" + str(epoch) + "_features.npy"
    test_known_known_label_path = known_feature_dir + "/features/test_known_known_" + str(epoch) + "_labels.npy"

    test_unknown_unknown_feature_path = unknown_feature_dir + "/test_unknown_unknown_features.npy"

    # Paths to save EVM model and all results
    save_result_dir = unknown_feature_dir



#####################################################
# Load the features and labels and check their shape
#####################################################
train_known_known_feature = np.load(train_known_known_feature_path)
train_known_known_label = np.load(train_known_known_label_path)

test_known_known_feature = np.load(test_known_known_feature_path)
test_known_known_label = np.load(test_known_known_label_path)

test_unknown_unknown_feature = np.load(test_unknown_unknown_feature_path)


if debug:
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
    test_known_known_feature = test_known_known_feature[:100]
    test_unknown_unknown_feature = test_unknown_unknown_feature[:100]

    train_known_known_label = train_known_known_label[:100]
    test_known_known_label = test_known_known_label[:100]

else:
    pass
    # TODO: for the real features, only consider exit 5


print("Train known known:", train_known_known_feature.shape, train_known_known_label.shape)
print("Test known known:", test_known_known_feature.shape, test_known_known_label.shape)
print("Test unknown unknown:", test_unknown_unknown_feature.shape)


####################################################
# TODO: PCA?? (TBD)
####################################################



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
    print(probs.shape) # Shape [nb_samples, nb_clfs, nb_classes]

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




def train_test_svm(train_known_feature,
                   train_known_labels,
                   test_known_feature,
                   test_known_labels,
                   test_unknown_feature):
    """

    :param train_known_feature:
    :param test_known_feature:
    :param test_unknown_feature:
    :param unknown_thresholds:
    :param debug:
    :return:
    """
    print("Training SVM...")
    svm_model = svm.SVC(kernel=kernel,
                        degree=degree,
                        C=c,
                        probability=True).fit(train_known_feature, train_known_labels)

    # Test SVM
    print("Testing SVM...")

    # For known_known classes, using predict is enough
    pred_known_known = svm_model.predict(test_known_feature)
    unknown_unknown_prob = svm_model.predict_proba(test_unknown_feature)

    acc_known_known = accuracy_score(test_known_labels, pred_known_known)
    print("Test known known accuracy:", acc_known_known)

    return acc_known_known, unknown_unknown_prob




def train_test_evm(train_known_feature,
                   train_known_labels,
                   test_known_feature,
                   test_unknown_feature,
                   tail_size):
    """

    :param train_known_feature:
    :param train_known_labels:
    :param test_known_feature:
    :param test_unknown_feature:
    :return:
    """

    labels = list(np.unique(train_known_labels))

    # Define EVM
    print("Creating EVM... Current tail size is %d" % tail_size)
    evm = ExtremeValueMachine(tail_size=tail_size,
            cover_threshold=0.5,
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
    print("Saving EVM model")
    evm.save(evm_model_save_path)

    print(test_known_feature.shape)


    # Test EVM
    print("Testing EVM")

    # Convert data type
    test_known_known_feature = torch.from_numpy(test_known_feature).float()
    test_unknown_unknown_feature = torch.from_numpy(test_unknown_feature).float()

    # Known known
    print("Testing known known")

    known_known_pred = evm.predict(test_known_known_feature)
    known_known_known_probs = evm.known_probs(test_known_known_feature)

    known_known_pred = known_known_pred.cpu().detach().numpy()
    known_known_known_probs = known_known_known_probs.cpu().detach().numpy()

    print("known_known_pred", known_known_pred.shape)
    print("known_known_known_probs", known_known_known_probs.shape)

    # unknown unknown
    unknown_unknown_pred = evm.predict(test_unknown_unknown_feature)
    unknown_unknown_known_probs = evm.known_probs(test_unknown_unknown_feature)
    unknown_unknown_pred = unknown_unknown_pred.cpu().detach().numpy()
    unknown_unknown_known_probs = unknown_unknown_known_probs.cpu().detach().numpy()

    print("unknown_unknown_pred", unknown_unknown_pred.shape)
    print("unknown_unknown_known_probs", unknown_unknown_known_probs.shape)

    return known_known_pred, known_known_known_probs, unknown_unknown_pred, unknown_unknown_known_probs




if __name__ == '__main__':
    # TODO: Get threshold for novelty


    # Option for svm
    if run_svm:
        acc_known_known, \
        unknown_unknown_prob = train_test_svm(train_known_feature=train_known_known_feature,
                                               train_known_labels=train_known_known_label,
                                               test_known_feature=test_known_known_feature,
                                               test_known_labels=test_known_known_label,
                                               test_unknown_feature=test_unknown_unknown_feature)

        # TODO: post-process for unknown

    # Option for evm
    if run_evm:
        for one_tail in tail_size:
            known_known_pred, \
            known_known_known_probs, \
            unknown_unknown_pred, \
            unknown_unknown_known_probs = train_test_evm(train_known_feature=train_known_known_feature,
                                                           train_known_labels=train_known_known_label,
                                                           test_known_feature=test_known_known_feature,
                                                           test_unknown_feature=test_unknown_unknown_feature,
                                                           tail_size=one_tail)

            # TODO: post-process for both known and unknown



