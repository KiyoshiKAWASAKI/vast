# Separating PCA from pipeline, do it once and save reduced features

import time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import svm, datasets
from sklearn.metrics import top_k_accuracy_score




####################################################
# Model and data paths
####################################################
# TODO: cross entropy
# model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_0", 147
model_dir, epoch = "2022-02-13/known_only_cross_entropy/seed_1", 181
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
print("Processing model: ", model_dir)

feature_base_scratch = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/msd_net"
feature_base_home = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/models/msd_net"

# Train and test feature path (with epoch index)
train_known_known_feature_path = feature_base_home + "/" + model_dir + \
                                 "/features/train_known_known_epoch_" + str(epoch) + "_features.npy"
train_known_known_label_path = feature_base_home + "/" + model_dir + \
                               "/features/train_known_known_epoch_" + str(epoch) + "_labels.npy"
train_known_known_feature = np.load(train_known_known_feature_path)
train_known_known_label = np.load(train_known_known_label_path)
print("Train feature loaded")

valid_known_known_feature_path = feature_base_home + "/" + model_dir + \
                                 "/features/valid_known_known_epoch_" + str(epoch) + "_features.npy"
valid_known_known_label_path = feature_base_home + "/" + model_dir + \
                               "/features/valid_known_known_epoch_" + str(epoch) + "_labels.npy"

valid_known_known_feature = np.load(valid_known_known_feature_path)
valid_known_known_label = np.load(valid_known_known_label_path)
print("Valid feature loaded")

test_known_known_feature_path_p0 = feature_base_scratch + "/" + model_dir + \
                                   "/test_results/test_known_known_epoch_" + str(epoch) + "_part_0_features.npy"
test_known_known_label_path_p0 = feature_base_scratch + "/" + model_dir + \
                                 "/test_results/test_known_known_epoch_" + str(epoch) + "_part_0_labels.npy"
test_known_known_feat_p0 = np.load(test_known_known_feature_path_p0)
test_known_known_labels_p0 = np.load(test_known_known_label_path_p0)
print("Test part 0 loaded")

test_known_known_feature_path_p1 = feature_base_scratch + "/" + model_dir + \
                                   "/test_results/test_known_known_epoch_" + str(epoch) + "_part_1_features.npy"
test_known_known_label_path_p1 = feature_base_scratch + "/" + model_dir + \
                                 "/test_results/test_known_known_epoch_" + str(epoch) + "_part_1_labels.npy"
test_known_known_feat_p1 = np.load(test_known_known_feature_path_p1)
test_known_known_labels_p1 = np.load(test_known_known_label_path_p1)
print("Test part 1 loaded")

test_known_known_feature_path_p2 = feature_base_scratch + "/" + model_dir + \
                                   "/test_results/test_known_known_epoch_" + str(epoch) + "_part_2_features.npy"
test_known_known_label_path_p2 = feature_base_scratch + "/" + model_dir + \
                                 "/test_results/test_known_known_epoch_" + str(epoch) + "_part_2_labels.npy"
test_known_known_feat_p2 = np.load(test_known_known_feature_path_p2)
test_known_known_labels_p2 = np.load(test_known_known_label_path_p2)
print("Test part 2 loaded")

test_known_known_feature_path_p3 = feature_base_scratch + "/" + model_dir + \
                                   "/test_results/test_known_known_epoch_" + str(epoch) + "_part_3_features.npy"
test_known_known_label_path_p3 = feature_base_scratch + "/" + model_dir + \
                                 "/test_results/test_known_known_epoch_" + str(epoch) + "_part_0_labels.npy"
test_known_known_feat_p3 = np.load(test_known_known_feature_path_p3)
test_known_known_labels_p3 = np.load(test_known_known_label_path_p3)
print("Test part 3 loaded")

test_unknown_unknown_feature_path = feature_base_home + "/" + model_dir + \
                                    "/test_results/unknown_unknown_epoch_" + str(epoch) + "_features.npy"
test_unknown_unknown_feature = np.load(test_unknown_unknown_feature_path)
print("Unknown unknown loaded")



####################################################
# PCA
####################################################
sc = StandardScaler()
pca = IncrementalPCA(n_components=70, batch_size=512)


########################################################################################################
train_feature_scaled = sc.fit_transform(train_known_known_feature)
train_feature_reduced = pca.fit_transform(train_feature_scaled)
np.save(feature_base_scratch + "/" + model_dir + "/test_results/train_known_known_epoch_" + str(epoch) + "_features_reduced.npy",
        train_feature_reduced)
print("train_feature_reduced", train_feature_reduced.shape)

########################################################################################################
valid_feature_scaled = sc.fit_transform(valid_known_known_feature)
valid_feature_reduced = pca.fit_transform(valid_feature_scaled)
np.save(feature_base_scratch + "/" + model_dir + "/test_results/valid_known_known_epoch_" + str(epoch) + "_features_reduced.npy",
        valid_feature_reduced)
print("valid_feature_reduced", valid_feature_reduced.shape)

########################################################################################################
test_known_feature_p0_scaled = sc.fit_transform(test_known_known_feat_p0)
test_known_feature_p0_reduced = pca.fit_transform(test_known_feature_p0_scaled)
np.save(feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_features_p0_reduced.npy",
        test_known_feature_p0_reduced)
print("test_known_feature_p0_reduced", test_known_feature_p0_reduced.shape)

test_known_feature_p1_scaled = sc.fit_transform(test_known_known_feat_p1)
test_known_feature_p1_reduced = pca.fit_transform(test_known_feature_p1_scaled)
np.save(feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_features_p1_reduced.npy",
        test_known_feature_p1_reduced)
print("test_known_feature_p1_reduced", test_known_feature_p1_reduced.shape)

test_known_feature_p2_scaled = sc.fit_transform(test_known_known_feat_p2)
test_known_feature_p2_reduced = pca.fit_transform(test_known_feature_p2_scaled)
np.save(feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_features_p2_reduced.npy",
        test_known_feature_p2_reduced)
print("test_known_feature_p2_reduced", test_known_feature_p2_reduced.shape)

test_known_feature_p3_scaled = sc.fit_transform(test_known_known_feat_p3)
test_known_feature_p3_reduced = pca.fit_transform(test_known_feature_p3_scaled)
np.save(feature_base_scratch + "/" + model_dir + "/test_results/test_known_known_epoch_" + str(epoch) + "_features_p3_reduced.npy",
        test_known_feature_p3_reduced)
print("test_known_feature_p3_reduced", test_known_feature_p3_reduced.shape)

########################################################################################################
test_unknown_feature_scaled = sc.fit_transform(test_unknown_unknown_feature)
test_unknown_feature_reduced = pca.fit_transform(test_unknown_feature_scaled)
np.save(feature_base_scratch + "/" + model_dir + "/test_results/test_unknown_unknown_epoch_" + str(epoch) + "_features_reduced.npy",
        test_unknown_feature_reduced)
print("test_unknown_feature_reduced", test_unknown_feature_reduced.shape)

