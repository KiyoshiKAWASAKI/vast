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
kernel = "poly"
degree = 3
c = 1

device = "cuda:0"

train_test_svm = True
train_test_evm = False

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

    # Paths to save results
    save_svm_txt_path = unknown_feature_dir + "/svm_result.txt"
    save_evm_txt_path = unknown_feature_dir + "/evm_result.txt"

    # Save paths for SVM
    unknown_unknown_prob_save_path = unknown_feature_dir + "/svm_test_ukuk_prob.npy"

    # Save probs from EVM
    evm_model_save_path = unknown_feature_dir + "/evm_model.pkl"

    known_known_pred_path = unknown_feature_dir + "/known_known_pred.npy"
    known_known_known_probs_path = unknown_feature_dir + "/known_known_known_probs.npy"

    unknown_unknown_pred_path = unknown_feature_dir + "/unknown_unknown_pred.npy"
    unknown_unknown_known_probs_path = unknown_feature_dir + "/unknown_unknown_known_probs.npy"


#####################################################
# Load the features and labels and check their shape
#####################################################
train_known_known_feature = np.load(train_known_known_feature_path)
train_known_known_label = np.load(train_known_known_label_path)

train_features = train_known_known_feature
train_labels = train_known_known_label

print("train_features", train_features.shape)
print("train_labels", train_labels.shape)

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



print("Train known known:", train_known_known_feature.shape, train_known_known_label.shape)
print("Test known known:", test_known_known_feature.shape, test_known_known_label.shape)
print("Test unknown unknown:", test_unknown_unknown_feature.shape)


####################################################
# TODO: PCA??
####################################################






####################################################
# SVM
####################################################
if train_test_svm:
    # Define a SVM using scikit-learn and train/fit it
    print("Training SVM...")

    if debug:
        features = train_known_known_feature
        labels = train_known_known_label
    else:
        features = train_features
        labels = train_labels


    svm_model = svm.SVC(kernel=kernel,
                        degree=degree,
                        C=c,
                        probability=True).fit(features, labels)

    # Test SVM
    print("Testing SVM...")

    # For known_known classes, using predict is enough
    pred_known_known = svm_model.predict(test_known_known_feature)
    acc_known_known = accuracy_score(test_known_known_label, pred_known_known)
    print("Test known known accuracy:", acc_known_known)

    # For known_unknown and unknown_unknown save the probability
    if debug:
        known_unknown_prob = svm_model.predict_proba(test_known_known_feature)
        unknown_unknown_prob = svm_model.predict_proba(test_known_known_feature)
        # print(known_unknown_prob) # 100*2 list (nb_sample*nb_class)

        known_unknown_prob_np = np.asarray(known_unknown_prob)
        print(known_unknown_prob_np.shape)

    else:
        unknown_unknown_prob = svm_model.predict_proba(test_unknown_unknown_feature)

        # save probability as npy
        np.save(unknown_unknown_prob, unknown_unknown_prob_save_path)


####################################################
# EVM
####################################################
if train_test_evm:
    # Define the labels for EVM
    if debug:
        labels = list(np.unique(train_known_known_label))

    else:
        labels = list(np.unique(train_labels))

    print("Labels", labels)

    # Define EVM
    print("Create EVM")
    evm = ExtremeValueMachine(tail_size=10,
            cover_threshold=0.5,
            distance_multiplier=1.0,
            labels=labels,
            distance_metric="cosine",
            chunk_size=200,
            device=device,
            tail_size_is_ratio=False)

    # Train(fit) EVM
    print("Training EVM")

    # if os.path.exists()
    # model_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/cvpr/" \
    #              "2021-10-24/cross_entropy_only/seed_0/evm_model.pkl"
    # evm.load(model_path)

    if debug:
        evm.fit(train_known_known_feature, train_known_known_label)
    else:
        evm.fit(train_features, train_labels)

    # Save EVM model, so no need to retrain
    print("Saving EVM model")
    evm.save(evm_model_save_path)

    # Test EVM
    print("Testing EVM")

    if debug:
        # For the debugging set, we only have known known
        test_known_known_feature = torch.from_numpy(test_known_known_feature).float()

        known_known_pred = evm.predict(test_known_known_feature)
        known_known_known_probs = evm.known_probs(test_known_known_feature)

        known_known_pred = known_known_pred.cpu().detach().numpy()
        known_known_known_probs = known_known_known_probs.cpu().detach().numpy()

        print("known_known_pred", known_known_pred.shape)
        print("known_known_known_probs", known_known_known_probs.shape)

    else:
        # Testing all 3 categories
        test_known_known_feature = torch.from_numpy(test_known_known_feature).float()
        test_unknown_unknown_feature = torch.from_numpy(test_unknown_unknown_feature).float()

        # Known known
        print("Testing known known")
        known_known_pred = evm.predict(test_known_known_feature)
        known_known_known_probs = evm.known_probs(test_known_known_feature)
        known_known_pred = known_known_pred.cpu().detach().numpy()
        known_known_known_probs = known_known_known_probs.cpu().detach().numpy()

        print(known_known_pred.shape)
        print(known_known_known_probs.shape)

        np.save(known_known_pred_path, known_known_pred)
        np.save(known_known_known_probs_path, known_known_known_probs)

        # unknown unknown
        unknown_unknown_pred = evm.predict(test_unknown_unknown_feature)
        unknown_unknown_known_probs = evm.known_probs(test_unknown_unknown_feature)
        unknown_unknown_pred = unknown_unknown_pred.cpu().detach().numpy()
        unknown_unknown_known_probs = unknown_unknown_known_probs.cpu().detach().numpy()

        print(unknown_unknown_pred.shape)
        print(unknown_unknown_known_probs.shape)

        np.save(unknown_unknown_pred_path, unknown_unknown_pred)
        np.save(unknown_unknown_known_probs_path, unknown_unknown_known_probs)

