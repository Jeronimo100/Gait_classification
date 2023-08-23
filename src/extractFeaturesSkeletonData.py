"""Example of Python code to solve Abnormal GAIT Problem using Skeleton data and inner-joint distance features"""

import math
import numpy as np
import trainModels

from numpy import save, load

loaded = np.load('../resources/DIRO_skeletons.npz')

# get skeleton data of size (n_subject, n_gait, n_frame, 25*3)
data = loaded['data']


# skeleton is array with dimension (1,75)
# result is feature vector of length 300 (inner-joints-distances) *** 25*24/2 = 300 ***
def get_singular_frame_feature_vector(skeleton):
    singular_frame_feature_vector = []
    # for 25 points
    for i in range(0, 25):
        for j in range(i + 1, 25):
            try:
                singular_frame_feature_vector.append(
                    math.dist((skeleton[3 * i], skeleton[3 * i + 1], skeleton[3 * i + 2]),
                              (skeleton[3 * j], skeleton[3 * j + 1], skeleton[3 * j + 2])))
            except IndexError:
                print("Check data, we should only have 25 joints")
    return singular_frame_feature_vector


# skeleton_frames is array with dimension (1200, 75)
# generate feature vector for every subsequence of the video
def get_multiple_frame_feature_vector(skeleton_frames, sliding_window_parameter):
    multiple_frame_feature_vector = np.empty((1200 - sliding_window_parameter, 300 * (sliding_window_parameter + 1)),
                                             float)
    for i in range(0, 1200 - sliding_window_parameter):
        temp_array = []
        temp_array += get_singular_frame_feature_vector(skeleton_frames[i])
        for j in range(0, sliding_window_parameter):
            temp_array += get_singular_frame_feature_vector(skeleton_frames[i + j + 1])
        multiple_frame_feature_vector[i] = temp_array
    return multiple_frame_feature_vector


# Gaits are array with dimension (9, 1200, 75)

# Result is (9 * (1200-sliding_window_parameter)) subsequences
# represented by (300 * sliding_window_parameter) features + 1 value for the label
# + 1 value for the frame sequence it belongs to
def get_all_gaits_labeled_for_one_person(gaits, sliding_window_parameter):
    # for i=0, which is normal walking gait, label 1
    temp_matrix = get_multiple_frame_feature_vector(gaits[0, :, :], sliding_window_parameter)
    temp_matrix = np.hstack((
        temp_matrix,
        np.hstack((np.ones((1200 - sliding_window_parameter, 1)),
                   np.full((1200 - sliding_window_parameter, 1), 0)))))
    # for i in [1,9], which are abnormal walking gait, label 0
    for i in range(1, 9):
        temp_matrix = np.vstack((temp_matrix,
                                 np.hstack((get_multiple_frame_feature_vector(gaits[i, :, :], sliding_window_parameter),
                                            np.hstack((np.zeros((1200 - sliding_window_parameter, 1)),
                                                       np.full((1200 - sliding_window_parameter, 1), i)))))))
    return temp_matrix


def get_partial_gaits_labeled_for_one_person(gaits, sliding_window_parameter):
    # for i=0, which is normal walking gait, label 1
    temp_matrix = get_multiple_frame_feature_vector(gaits[0, :, :], sliding_window_parameter)
    temp_matrix = np.hstack((
        temp_matrix,
        np.hstack((np.ones((1200 - sliding_window_parameter, 1)),
                   np.full((1200 - sliding_window_parameter, 1), 0)))))

    # if we want to only analyze part of the data to balance out classes
    # for 'i' is 3, 4, 7 or 8, which are abnormal walking gait, label 0
    for i in [3]:
        temp_matrix = np.vstack((temp_matrix,
                                 np.hstack((get_multiple_frame_feature_vector(gaits[i, :, :], sliding_window_parameter),
                                            np.hstack((np.zeros((1200 - sliding_window_parameter, 1)),
                                                       np.full((1200 - sliding_window_parameter, 1), i)))))))
    return temp_matrix


def get_all_gaits_labeled_for_all_persons(full_data, sliding_window_parameter):
    temp_matrix = get_all_gaits_labeled_for_one_person(full_data[0, :, :, :], sliding_window_parameter)
    for i in range(1, 9):
        temp_to_change_sequence_id = get_all_gaits_labeled_for_one_person(full_data[i, :, :, :],
                                                                          sliding_window_parameter)
        temp_to_change_sequence_id[:, -1] += (i * 9)
        temp_matrix = np.vstack((temp_matrix,
                                 temp_to_change_sequence_id))
    return temp_matrix


def get_partial_gaits_labeled_for_all_persons(full_data, sliding_window_parameter):
    temp_matrix = get_partial_gaits_labeled_for_one_person(full_data[0, :, :, :], sliding_window_parameter)
    for i in range(1, 9):
        temp_to_change_sequence_id = get_partial_gaits_labeled_for_one_person(full_data[i, :, :, :],
                                                                              sliding_window_parameter)
        temp_to_change_sequence_id[:, -1] += (i * 9)
        temp_matrix = np.vstack((temp_matrix,
                                 temp_to_change_sequence_id))
    return temp_matrix


# test singular frame feature vector
def test_singular_frame_feature_vector():
    temp = get_singular_frame_feature_vector(data[0, 0, 0, :])
    print(len(temp))
    print(temp[0:5])


# test multiple frames feature vector
def test_multiple_frame_feature_vector():
    sliding_window_param = 10
    temp = get_multiple_frame_feature_vector(data[0, 0, :, :], sliding_window_param)
    print(temp.shape)
    print(temp[0:5])


# test sliding window features
def test_all_gaits_featured_and_labeled():
    sliding_window_param = 10
    all_gaits = data[0, :, :, :]
    print(all_gaits.shape)
    gaits_featured_and_labeled = get_all_gaits_labeled_for_one_person(all_gaits, sliding_window_param)
    print(gaits_featured_and_labeled.shape)
    print(gaits_featured_and_labeled[0:4])


def test_data_featured_and_labeled():
    sliding_window_param = 0
    all_features = get_all_gaits_labeled_for_all_persons(data, sliding_window_param)
    print(all_features.shape)
    print(np.unique(all_features[:, -1]))


def compute_and_save_features_partial_data(sliding_window_param):
    features_vector = get_partial_gaits_labeled_for_all_persons(data, sliding_window_param)
    save("../feature_vectors/feature_vector_{sliding_window}_partial.npy".format(sliding_window=sliding_window_param),
         features_vector)
    return features_vector


def compute_and_save_features(sliding_window_param):
    features_vector = get_all_gaits_labeled_for_all_persons(data, sliding_window_param)
    save("../feature_vectors/feature_vector_{sliding_window}.npy".format(sliding_window=sliding_window_param),
         features_vector)
    return features_vector


def create_subsequence_features():
    for i in range(0, 11):
        print("Getting features vector for sliding window {sliding_window}".format(sliding_window=i))
        features_vector = get_all_gaits_labeled_for_all_persons(data, i)
        save("../feature_vectors/feature_vector_{sliding_window}.npy".format(sliding_window=i), features_vector)

        # if exists("../feature_vectors/feature_vector_{sliding_window}.npy".format(sliding_window=i)):


# get the right feature vectors according to sliding window
def get_feature_vector(sliding_window_param):
    try:
        feature_input = load("../feature_vectors/feature_vector_{sliding_window}.npy"
                             .format(sliding_window=sliding_window_param))
    except FileNotFoundError:
        print("Pre-computed feature vectors not found, start computing now...")
        feature_input = compute_and_save_features(sliding_window_param)

    print("Acquired input feature vectors for sliding window {sliding_window} (and saved as file in "
          "../feature_vectors/) "
          .format(sliding_window=sliding_window_param))
    return feature_input


# get the right feature vectors according to sliding window
def get_feature_vector_partial(sliding_window_param):
    try:
        feature_input = load("../feature_vectors/feature_vector_{sliding_window}_partial.npy"
                             .format(sliding_window=sliding_window_param))
    except FileNotFoundError:
        print("Pre-computed feature vectors not found, start computing now...")
        feature_input = compute_and_save_features_partial_data(sliding_window_param)

    print("Acquired input feature vectors for sliding window {sliding_window} (and saved as file in "
          "../feature_vectors/) "
          .format(sliding_window=sliding_window_param))
    return feature_input


def run_on_entire_data(sliding_window_param):
    feature_input = get_feature_vector(sliding_window_param)

    # splitting according to slightly changed default separation
    # training_set = {subject 1, subject 3, subject 5, subject 6, subject 9}
    # test_set = {subject 2, subject 4, subject 7, subject 8}
    subject_1 = feature_input[0:(9 * (1200 - sliding_window_param)), :]
    subject_2 = feature_input[(9 * (1200 - sliding_window_param)):2 * (9 * (1200 - sliding_window_param)), :]
    subject_3 = feature_input[2 * (9 * (1200 - sliding_window_param)):3 * (9 * (1200 - sliding_window_param)), :]
    subject_4 = feature_input[3 * (9 * (1200 - sliding_window_param)):4 * (9 * (1200 - sliding_window_param)), :]
    subject_5 = feature_input[4 * (9 * (1200 - sliding_window_param)):5 * (9 * (1200 - sliding_window_param)), :]
    subject_6 = feature_input[5 * (9 * (1200 - sliding_window_param)):6 * (9 * (1200 - sliding_window_param)), :]
    subject_7 = feature_input[6 * (9 * (1200 - sliding_window_param)):7 * (9 * (1200 - sliding_window_param)), :]
    subject_8 = feature_input[7 * (9 * (1200 - sliding_window_param)):8 * (9 * (1200 - sliding_window_param)), :]
    subject_9 = feature_input[8 * (9 * (1200 - sliding_window_param)):9 * (9 * (1200 - sliding_window_param)), :]

    training_data = np.vstack((subject_1, subject_3, subject_5, subject_6, subject_8, subject_9))
    test_data = np.vstack((subject_2, subject_4, subject_7))

    # split data and train models on subsequences
    x_train = np.delete(training_data, -2, axis=1)
    y_train = training_data[:, -2]
    x_test = np.delete(test_data, -2, axis=1)
    y_test = test_data[:, -2]

    x_train = x_train[:, :-1]

    x_test_sequence_label = x_test[:, -1]
    x_test = x_test[:, :-1]

    # runs and test for subsequences
    y_pred_dec_tree = trainModels.run_dec_tree(x_train, x_test, y_train, x_test_sequence_label, sliding_window_param,
                                               "")
    y_pred_bayes = trainModels.run_bayes(x_train, x_test, y_train, x_test_sequence_label, sliding_window_param, "")
    y_pred_svm = trainModels.run_svm(x_train, x_test, y_train, x_test_sequence_label, sliding_window_param, "")
    y_pred_rand_forest = trainModels.run_random_forest(x_train, x_test, y_train, x_test_sequence_label,
                                                       sliding_window_param, "")
    y_pred_log_reg = trainModels.run_log_reg(x_train, x_test, y_train, x_test_sequence_label, sliding_window_param, "")
    y_pred_mlp_class = trainModels.run_mlp_classifier(x_train, x_test, y_train, x_test_sequence_label,
                                                      sliding_window_param, "")

    return y_test, y_pred_dec_tree, y_pred_bayes, y_pred_svm, y_pred_rand_forest, y_pred_log_reg, y_pred_mlp_class


# to go against 8/1 partition of negative/positive class labels
def run_on_partial(sliding_window_param):
    feature_input = get_feature_vector_partial(sliding_window_param)

    # splitting according to slightly changed default separation, only 2 gaits
    # training_set = {subject 1, subject 3, subject 5, subject 6, subject 9}
    # test_set = {subject 2, subject 4, subject 7, subject 8}
    subject_1 = feature_input[0:(2 * (1200 - sliding_window_param)), :]
    subject_2 = feature_input[(2 * (1200 - sliding_window_param)):2 * (2 * (1200 - sliding_window_param)), :]
    subject_3 = feature_input[2 * (2 * (1200 - sliding_window_param)):3 * (2 * (1200 - sliding_window_param)), :]
    subject_4 = feature_input[3 * (2 * (1200 - sliding_window_param)):4 * (2 * (1200 - sliding_window_param)), :]
    subject_5 = feature_input[4 * (2 * (1200 - sliding_window_param)):5 * (2 * (1200 - sliding_window_param)), :]
    subject_6 = feature_input[5 * (2 * (1200 - sliding_window_param)):6 * (2 * (1200 - sliding_window_param)), :]
    subject_7 = feature_input[6 * (2 * (1200 - sliding_window_param)):7 * (2 * (1200 - sliding_window_param)), :]
    subject_8 = feature_input[7 * (2 * (1200 - sliding_window_param)):8 * (2 * (1200 - sliding_window_param)), :]
    subject_9 = feature_input[8 * (2 * (1200 - sliding_window_param)):9 * (2 * (1200 - sliding_window_param)), :]

    training_data = np.vstack((subject_1, subject_3, subject_5, subject_6, subject_8, subject_9))
    test_data = np.vstack((subject_2, subject_4, subject_7))

    # split data and train models on subsequences
    x_train = np.delete(training_data, -2, axis=1)
    y_train = training_data[:, -2]
    x_test = np.delete(test_data, -2, axis=1)
    y_test = test_data[:, -2]

    x_train = x_train[:, :-1]

    x_test_sequence_label = x_test[:, -1]
    x_test = x_test[:, :-1]

    # runs and test for subsequences
    y_pred_dec_tree = trainModels.run_dec_tree(x_train, x_test, y_train, x_test_sequence_label,
                                               sliding_window_param, "partial")
    y_pred_bayes = trainModels.run_bayes(x_train, x_test, y_train, x_test_sequence_label, sliding_window_param,
                                         "partial")
    y_pred_svm = trainModels.run_svm(x_train, x_test, y_train, x_test_sequence_label, sliding_window_param, "partial")
    y_pred_rand_forest = trainModels.run_random_forest(x_train, x_test, y_train, x_test_sequence_label,
                                                       sliding_window_param, "partial")
    y_pred_log_reg = trainModels.run_log_reg(x_train, x_test, y_train, x_test_sequence_label, sliding_window_param,
                                             "partial")
    y_pred_mlp_class = trainModels.run_mlp_classifier(x_train, x_test, y_train, x_test_sequence_label,
                                                      sliding_window_param,
                                                      "partial")

    return y_test, y_pred_dec_tree, y_pred_bayes, y_pred_svm, y_pred_rand_forest, y_pred_log_reg, y_pred_mlp_class
