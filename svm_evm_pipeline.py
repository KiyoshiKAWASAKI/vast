# Training and testing SVM and EVM using generated network features

import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import svm, datasets
from sklearn.metrics import top_k_accuracy_score

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
percent = 50
pca_ratio = 0.99

device = "cuda:0"

run_svm = False
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

    feature_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/msd_net"

    known_feature_dir = os.path.join(feature_base, model_dir)
    unknown_feature_dir = os.path.join(feature_base, model_dir)

    # Train and test feature path (with epoch index)
    train_known_known_feature_path = known_feature_dir + "/features/train_known_known_epoch_" + str(epoch) + "_features.npy"
    train_known_known_label_path = known_feature_dir + "/features/train_known_known_epoch_" + str(epoch) + "_labels.npy"

    valid_known_known_feature_path = known_feature_dir + "/features/valid_known_known_epoch_" + str(epoch) + "_features.npy"
    valid_known_known_label_path = known_feature_dir + "/features/valid_known_known_epoch_" + str(epoch) + "_labels.npy"
    valid_known_known_probs_path = known_feature_dir + "/features/valid_known_known_epoch_" + str(epoch) + "_probs.npy"

    valid_known_known_feature = np.load(valid_known_known_feature_path)
    valid_known_known_label = np.load(valid_known_known_label_path)

    test_known_known_feature_path_p0 = known_feature_dir + "/test_known_known_epoch_" + str(epoch) + "_part_0_features.npy"
    test_known_known_label_path_p0 = known_feature_dir + "/test_known_known_epoch_" + str(epoch) + "_part_0_labels.npy"
    test_known_known_feat_p0 = np.load(test_known_known_feature_path_p0)
    test_known_known_labels_p0 = np.load(test_known_known_label_path_p0)

    test_known_known_feature_path_p1 = known_feature_dir + "/test_known_known_epoch_" + str(epoch) + "_part_1_features.npy"
    test_known_known_label_path_p1 = known_feature_dir + "/test_known_known_epoch_" + str(epoch) + "_part_1_labels.npy"
    test_known_known_feat_p1 = np.load(test_known_known_feature_path_p1)
    test_known_known_labels_p1 = np.load(test_known_known_label_path_p1)

    test_known_known_feature_path_p2 = known_feature_dir + "/test_known_known_epoch_" + str(epoch) + "_part_2_features.npy"
    test_known_known_label_path_p2 = known_feature_dir + "/test_known_known_epoch_" + str(epoch) + "_part_2_labels.npy"
    test_known_known_feat_p2 = np.load(test_known_known_feature_path_p2)
    test_known_known_labels_p2 = np.load(test_known_known_label_path_p2)

    test_known_known_feature_path_p3 = known_feature_dir + "/test_known_known_epoch_" + str(epoch) + "_part_3_features.npy"
    test_known_known_label_path_p3 = known_feature_dir + "/test_known_known_epoch_" + str(epoch) + "_part_0_labels.npy"
    test_known_known_feat_p3 = np.load(test_known_known_feature_path_p3)
    test_known_known_labels_p3 = np.load(test_known_known_label_path_p3)

    test_unknown_unknown_feature_path = unknown_feature_dir + "/test_unknown_unknown_epoch_" + str(epoch) +"_features.npy"

    test_known_known_feature = np.concatenate((test_known_known_feat_p0, test_known_known_feat_p1,
                                             test_known_known_feat_p2, test_known_known_feat_p3), axis=0)
    test_known_known_label = np.concatenate((test_known_known_labels_p0, test_known_known_labels_p1,
                                              test_known_known_labels_p2, test_known_known_labels_p3), axis=0)

    # Paths to save EVM model, EVM probs and results
    save_result_dir = unknown_feature_dir


#####################################################
# Load the features and labels and check their shape
#####################################################
train_known_known_feature = np.load(train_known_known_feature_path)
train_known_known_label = np.load(train_known_known_label_path)
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
    train_known_known_feature = train_known_known_feature[:500]
    valid_known_known_feature = train_known_known_feature
    test_known_known_feature = test_known_known_feature[:30]
    test_unknown_unknown_feature = test_unknown_unknown_feature[:100]

    train_known_known_label = train_known_known_label[:500]
    valid_known_known_label = train_known_known_label
    test_known_known_label = test_known_known_label[:30]

else:
    pass


print("Train known known:", train_known_known_feature.shape, train_known_known_label.shape)
print("Valid known known", valid_known_known_feature.shape, valid_known_known_label.shape)
print("Test known known:", test_known_known_feature.shape, test_known_known_label.shape)
print("Test unknown unknown:", test_unknown_unknown_feature.shape)


####################################################
# TODO: PCA
####################################################
sc = StandardScaler()
pca = PCA(n_components=pca_ratio)

train_feature_scaled = sc.fit_transform(train_known_known_feature)
valid_feature_scaled = sc.fit_transform(valid_known_known_feature)
test_known_feature_scaled = sc.fit_transform(test_known_known_feature)
test_unknown_feature_scaled = sc.fit_transform(test_unknown_unknown_feature)

train_feature_reduced = pca.fit_transform(train_feature_scaled)
valid_feature_reduced = pca.fit_transform(valid_feature_scaled)
test_known_feature_reduced = pca.fit_transform(test_known_feature_scaled)
test_unknown_feature_reduced = pca.fit_transform(test_unknown_feature_scaled)

print("train_feature_reduced", train_feature_reduced.shape)
print("valid_feature_reduced", valid_feature_reduced.shape)
print("test_known_feature_reduced", test_known_feature_reduced.shape)
print("test_unknown_feature_reduced", test_unknown_feature_reduced.shape)


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
def get_known_acc(known_probs):
    """
    top-1, top-3, top-5 acc for evm

    :param known_probs:
    :return:
    """




def train_test_svm(train_known_feature,
                   train_known_labels,
                   valid_known_feature,
                   valid_known_labels,
                   test_known_feature,
                   test_known_labels,
                   test_unknown_feature,
                   debug):
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
    svm_model = svm.SVC(kernel=kernel,
                        degree=degree,
                        C=c,
                        probability=True).fit(train_known_feature, train_known_labels)

    # Test SVM
    print("Testing SVM...")

    # For known_known classes, using predict is enough
    pred_valid = svm_model.predict(valid_known_feature)
    pred_known_known = svm_model.predict(test_known_feature)
    unknown_unknown_prob = svm_model.predict_proba(test_unknown_feature)


    if debug:
        valid_acc_known_top_1 = top_k_accuracy_score(valid_known_labels, pred_valid, k=1)
        test_acc_known_top_1 = top_k_accuracy_score(test_known_labels, pred_known_known, k=1)

        return valid_acc_known_top_1, 0.000, 0.000, \
               test_acc_known_top_1, 0.000, 0.000, \
               unknown_unknown_prob

    else:
        valid_acc_known_top_1 = top_k_accuracy_score(valid_known_labels, pred_valid, k=1)
        valid_acc_known_top_3 = top_k_accuracy_score(valid_known_labels, pred_valid, k=3)
        valid_acc_known_top_5 = top_k_accuracy_score(valid_known_labels, pred_valid, k=5)

        test_acc_known_top_1 = top_k_accuracy_score(test_known_labels, pred_known_known, k=1)
        test_acc_known_top_3 = top_k_accuracy_score(test_known_labels, pred_known_known, k=3)
        test_acc_known_top_5 = top_k_accuracy_score(test_known_labels, pred_known_known, k=5)

        return valid_acc_known_top_1, valid_acc_known_top_3, valid_acc_known_top_5,\
               test_acc_known_top_1, test_acc_known_top_3, test_acc_known_top_5, \
               unknown_unknown_prob




def train_test_evm(train_known_feature,
                   train_known_labels,
                   valid_known_feature,
                   valid_known_labels,
                   test_known_feature,
                   test_known_labels,
                   test_unknown_feature,
                   tail_size,
                   debug):
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
    evm_model_save_path = save_result_dir + "/evm_model_tail_size_" + str(tail_size) + ".pkl"
    print("Saving EVM model")
    evm.save(evm_model_save_path)

    # print(test_known_feature.shape)

    # Convert data type
    valid_known_known_feature = torch.from_numpy(valid_known_feature).float()
    test_known_known_feature = torch.from_numpy(test_known_feature).float()
    test_unknown_unknown_feature = torch.from_numpy(test_unknown_feature).float()

    # Test EVM
    print("Testing EVM")
    valid_known_known_known_probs = evm.known_probs(valid_known_known_feature).cpu().detach().numpy()
    test_known_known_known_probs = evm.known_probs(test_known_known_feature).cpu().detach().numpy()
    unknown_unknown_known_probs = evm.known_probs(test_unknown_unknown_feature).cpu().detach().numpy()

    # Get predictions
    valid_pred_labels = np.argmax(valid_known_known_known_probs, axis=1)
    test_pred_labels = np.argmax(test_known_known_known_probs, axis=1)

    # Known accuracy
    if debug:
        valid_acc_known_top_1 = top_k_accuracy_score(valid_known_labels, valid_pred_labels, k=1)
        test_acc_known_top_1 = top_k_accuracy_score(test_known_labels, test_pred_labels, k=1)

        return valid_acc_known_top_1, 0.000, 0.000,\
               test_acc_known_top_1, 0.000, 0.000, \
               test_known_known_known_probs, unknown_unknown_known_probs

    else:
        valid_acc_known_top_1 = top_k_accuracy_score(valid_known_labels, valid_pred_labels, k=1)
        valid_acc_known_top_3 = top_k_accuracy_score(valid_known_labels, valid_pred_labels, k=3)
        valid_acc_known_top_5 = top_k_accuracy_score(valid_known_labels, valid_pred_labels, k=5)

        test_acc_known_top_1 = top_k_accuracy_score(test_known_labels, test_pred_labels, k=1)
        test_acc_known_top_3 = top_k_accuracy_score(test_known_labels, test_pred_labels, k=3)
        test_acc_known_top_5 = top_k_accuracy_score(test_known_labels, test_pred_labels, k=5)

        return valid_acc_known_top_1, valid_acc_known_top_3,  valid_acc_known_top_5, \
               test_acc_known_top_1, test_acc_known_top_3, test_acc_known_top_5, \
               valid_known_known_known_probs, test_known_known_known_probs, unknown_unknown_known_probs




if __name__ == '__main__':
    # Get threshold for novelty
    novelty_thresh = get_thresholds(npy_file_path=valid_known_known_probs_path,
                                    percentile=percent)

    # Option for svm
    if run_svm:
        svm_valid_top_1, svm_valid_top_3, \
        svm_valid_top_5, svm_test_top_1, \
        svm_test_top_3, svm_test_top_5, unknown_unknown_prob = train_test_svm(train_known_feature=train_feature_reduced,
                                                                              train_known_labels=train_known_known_label,
                                                                              valid_known_feature=valid_feature_reduced,
                                                                              valid_known_labels=valid_known_known_label,
                                                                               test_known_feature=test_known_feature_reduced,
                                                                               test_known_labels=test_known_known_label,
                                                                               test_unknown_feature=test_unknown_feature_reduced,
                                                                               debug=debug)

        # SVM post-process for unknown
        svm_acc_unknown_unknown = get_unknown_acc(probs=unknown_unknown_prob,
                                                  threshold=novelty_thresh[-1])

        # Save svm results
        save_svm_result_path = save_result_dir + "/svm_" + kernel + "_degree_" + str(degree) + \
                               "_c_" + str(c) + ".txt"

        print(svm_valid_top_1, svm_valid_top_3, svm_valid_top_5,
              svm_test_top_1, svm_test_top_3, svm_test_top_5,
              svm_acc_unknown_unknown)

        with open(save_svm_result_path, 'a') as f:
            f.write('%0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, %0.6f, ' %
                    (svm_valid_top_1, svm_valid_top_3, svm_valid_top_5,
                     svm_test_top_1, svm_test_top_3, svm_test_top_5, svm_acc_unknown_unknown))

    # Option for evm
    if run_evm:
        for one_tail in tail_size:
            evm_acc_top1, evm_acc_top3, evm_acc_top5, \
            known_known_known_probs, \
            unknown_unknown_known_probs = train_test_evm(train_known_feature=train_feature_reduced,
                                                           train_known_labels=train_known_known_label,
                                                           test_known_feature=test_known_feature_reduced,
                                                           test_unknown_feature=test_unknown_feature_reduced,
                                                           tail_size=one_tail,
                                                           debug=debug)

            # EVM post-process for unknown
            evm_acc_unknown_unknown = get_unknown_acc(probs=unknown_unknown_known_probs,
                                                      threshold=novelty_thresh[-1])

            # Save EVM results
            save_evm_result_path = save_result_dir + "/evm_result_tail_size_" + str(one_tail) + ".txt"

            with open(save_evm_result_path, 'a') as f:
                f.write('%0.6f, %0.6f, %0.6f, %0.6f, ' % (evm_acc_top1, evm_acc_top3,
                                                          evm_acc_top5, evm_acc_unknown_unknown))

            # Save EVM probs (for potential future use)
            save_known_prob_path = save_result_dir + "/evm_known_known_known_probs_tail_size" + str(one_tail) + ".npy"
            save_unknown_prob_path = save_result_dir + "/evm_unknown_unknown_known_probs_tail_size" + str(one_tail) + ".npy"

            np.save(save_known_prob_path, known_known_known_probs)
            np.save(save_unknown_prob_path, unknown_unknown_known_probs)

