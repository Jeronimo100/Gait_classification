import extractFeaturesSkeletonData as asd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def test_different_sliding_window():

    # values in between become too large to be processed by Apple M1 Pro Chip in reasonable time
    # window_size 10 is already 2GB and window_size 20 4GB of features
    window_sizes = [0, 5, 10, 20, 1189, 1199]

    for window_size in window_sizes:

        file = open("Stat_results.txt", 'a')

        print("computing stats for window size {}".format(window_size))
        y_true, y_pred_dec_tree, y_pred_bayes, y_pred_svm, y_pred_rand_forest, y_pred_log_reg, y_pred_mlp_class \
            = asd.run_on_entire_data(window_size)

        acc_dec_tree = accuracy_score(y_true, y_pred_dec_tree)
        f1_dec_tree = f1_score(y_true, y_pred_dec_tree)
        precision_dec_tree = precision_score(y_true, y_pred_dec_tree)
        recall_dec_tree = recall_score(y_true, y_pred_dec_tree)
        roc_auc_dec_tree = roc_auc_score(y_true, y_pred_dec_tree)

        acc_bayes = accuracy_score(y_true, y_pred_bayes)
        f1_bayes = f1_score(y_true, y_pred_bayes)
        precision_bayes = precision_score(y_true, y_pred_bayes)
        recall_bayes = recall_score(y_true, y_pred_bayes)
        roc_auc_bayes = roc_auc_score(y_true, y_pred_bayes)

        acc_svm = accuracy_score(y_true, y_pred_svm)
        f1_svm = f1_score(y_true, y_pred_svm)
        precision_svm = precision_score(y_true, y_pred_svm)
        recall_svm = recall_score(y_true, y_pred_svm)
        roc_auc_svm = roc_auc_score(y_true, y_pred_svm)

        acc_rand_forest = accuracy_score(y_true, y_pred_rand_forest)
        f1_rand_forest = f1_score(y_true, y_pred_rand_forest)
        precision_rand_forest = precision_score(y_true, y_pred_rand_forest)
        recall_rand_forest = recall_score(y_true, y_pred_rand_forest)
        roc_auc_rand_forest = roc_auc_score(y_true, y_pred_rand_forest)

        acc_log_reg = accuracy_score(y_true, y_pred_log_reg)
        f1_log_reg = f1_score(y_true, y_pred_log_reg)
        precision_log_reg = precision_score(y_true, y_pred_log_reg)
        recall_log_reg = recall_score(y_true, y_pred_log_reg)
        roc_auc_log_reg = roc_auc_score(y_true, y_pred_log_reg)

        acc_nn = accuracy_score(y_true, y_pred_mlp_class)
        f1_nn = f1_score(y_true, y_pred_mlp_class)
        precision_nn = precision_score(y_true, y_pred_mlp_class)
        recall_nn = recall_score(y_true, y_pred_mlp_class)
        roc_auc_nn = roc_auc_score(y_true, y_pred_mlp_class)

        file.write("##################################" + "\n")
        file.write("Stats for sliding window size: " + str(window_size) + "\n\n")

        file.write("Decision tree stats")
        file.write("Accuracy of decision trees: " + str(acc_dec_tree) + "\n")
        file.write("Precision of decision trees: " + str(precision_dec_tree) + "\n")
        file.write("Recall of decision trees: " + str(recall_dec_tree) + "\n")
        file.write("F1-score of decision trees: " + str(f1_dec_tree) + "\n")
        file.write("RocAuc score of decision trees: " + str(roc_auc_dec_tree) + "\n\n")

        file.write("Naive Bayes stats")
        file.write("Accuracy of naive bayes: " + str(acc_bayes) + "\n")
        file.write("Precision of naive bayes: " + str(precision_bayes) + "\n")
        file.write("Recall of naive bayes: " + str(recall_bayes) + "\n")
        file.write("F1-score of naive bayes: " + str(f1_bayes) + "\n")
        file.write("RocAuc score of naive bayes: " + str(roc_auc_bayes) + "\n\n")

        file.write("SVM stats")
        file.write("Accuracy of SVM: " + str(acc_svm) + "\n")
        file.write("Precision of SVM: " + str(precision_svm) + "\n")
        file.write("Recall of SVM: " + str(recall_svm) + "\n")
        file.write("F1-score of SVM: " + str(f1_svm) + "\n")
        file.write("RocAuc score of SVM: " + str(roc_auc_svm) + "\n\n")

        file.write("Random Forest stats")
        file.write("Accuracy of Random Forest: " + str(acc_rand_forest) + "\n")
        file.write("Precision of Random Forest: " + str(precision_rand_forest) + "\n")
        file.write("Recall of Random Forest: " + str(recall_rand_forest) + "\n")
        file.write("F1-score of Random Forest: " + str(f1_rand_forest) + "\n")
        file.write("RocAuc score of Random Forest: " + str(roc_auc_rand_forest) + "\n\n")

        file.write("Logistic Regression stats")
        file.write("Accuracy of Logistic Regression: " + str(acc_log_reg) + "\n")
        file.write("Precision of Logistic Regression: " + str(precision_log_reg) + "\n")
        file.write("Recall of Logistic Regression: " + str(recall_log_reg) + "\n")
        file.write("F1-score of Logistic Regression: " + str(f1_log_reg) + "\n")
        file.write("RocAuc score of Logistic Regression: " + str(roc_auc_log_reg) + "\n\n")

        file.write("Neural Networks stats")
        file.write("Accuracy of Neural Networks: " + str(acc_nn) + "\n")
        file.write("Precision of Neural Networks: " + str(precision_nn) + "\n")
        file.write("Recall of Neural Networks: " + str(recall_nn) + "\n")
        file.write("F1-score of Neural Networks: " + str(f1_nn) + "\n")
        file.write("RocAuc score of Neural Networks: " + str(roc_auc_nn) + "\n\n")

        file.close()


def test_different_sliding_window_partial():
    # values in between become too large to be processed by Apple M1 Pro Chip in reasonable time
    window_sizes = [0, 5, 10, 20, 1189, 1199]

    for window_size in window_sizes:
        file = open("Stat_results_partial.txt", 'a')

        print("computing stats for window size {}".format(window_size))
        y_true, y_pred_dec_tree, y_pred_bayes, y_pred_svm, y_pred_rand_forest, y_pred_log_reg, y_pred_mlp_class \
            = asd.run_on_partial(window_size)

        acc_dec_tree = accuracy_score(y_true, y_pred_dec_tree)
        f1_dec_tree = f1_score(y_true, y_pred_dec_tree)
        precision_dec_tree = precision_score(y_true, y_pred_dec_tree)
        recall_dec_tree = recall_score(y_true, y_pred_dec_tree)
        roc_auc_dec_tree = roc_auc_score(y_true, y_pred_dec_tree)

        acc_bayes = accuracy_score(y_true, y_pred_bayes)
        f1_bayes = f1_score(y_true, y_pred_bayes)
        precision_bayes = precision_score(y_true, y_pred_bayes)
        recall_bayes = recall_score(y_true, y_pred_bayes)
        roc_auc_bayes = roc_auc_score(y_true, y_pred_bayes)

        acc_svm = accuracy_score(y_true, y_pred_svm)
        f1_svm = f1_score(y_true, y_pred_svm)
        precision_svm = precision_score(y_true, y_pred_svm)
        recall_svm = recall_score(y_true, y_pred_svm)
        roc_auc_svm = roc_auc_score(y_true, y_pred_svm)

        acc_rand_forest = accuracy_score(y_true, y_pred_rand_forest)
        f1_rand_forest = f1_score(y_true, y_pred_rand_forest)
        precision_rand_forest = precision_score(y_true, y_pred_rand_forest)
        recall_rand_forest = recall_score(y_true, y_pred_rand_forest)
        roc_auc_rand_forest = roc_auc_score(y_true, y_pred_rand_forest)

        acc_log_reg = accuracy_score(y_true, y_pred_log_reg)
        f1_log_reg = f1_score(y_true, y_pred_log_reg)
        precision_log_reg = precision_score(y_true, y_pred_log_reg)
        recall_log_reg = recall_score(y_true, y_pred_log_reg)
        roc_auc_log_reg = roc_auc_score(y_true, y_pred_log_reg)

        acc_nn = accuracy_score(y_true, y_pred_mlp_class)
        f1_nn = f1_score(y_true, y_pred_mlp_class)
        precision_nn = precision_score(y_true, y_pred_mlp_class)
        recall_nn = recall_score(y_true, y_pred_mlp_class)
        roc_auc_nn = roc_auc_score(y_true, y_pred_mlp_class)

        file.write("##################################" + "\n")
        file.write("Stats for sliding window size: " + str(window_size) + "\n\n")

        file.write("Decision tree stats")
        file.write("Accuracy of decision trees: " + str(acc_dec_tree) + "\n")
        file.write("Precision of decision trees: " + str(precision_dec_tree) + "\n")
        file.write("Recall of decision trees: " + str(recall_dec_tree) + "\n")
        file.write("F1-score of decision trees: " + str(f1_dec_tree) + "\n")
        file.write("RocAuc score of decision trees: " + str(roc_auc_dec_tree) + "\n\n")

        file.write("Naive Bayes stats")
        file.write("Accuracy of naive bayes: " + str(acc_bayes) + "\n")
        file.write("Precision of naive bayes: " + str(precision_bayes) + "\n")
        file.write("Recall of naive bayes: " + str(recall_bayes) + "\n")
        file.write("F1-score of naive bayes: " + str(f1_bayes) + "\n")
        file.write("RocAuc score of naive bayes: " + str(roc_auc_bayes) + "\n\n")

        file.write("SVM stats")
        file.write("Accuracy of SVM: " + str(acc_svm) + "\n")
        file.write("Precision of SVM: " + str(precision_svm) + "\n")
        file.write("Recall of SVM: " + str(recall_svm) + "\n")
        file.write("F1-score of SVM: " + str(f1_svm) + "\n")
        file.write("RocAuc score of SVM: " + str(roc_auc_svm) + "\n\n")

        file.write("Random Forest stats")
        file.write("Accuracy of Random Forest: " + str(acc_rand_forest) + "\n")
        file.write("Precision of Random Forest: " + str(precision_rand_forest) + "\n")
        file.write("Recall of Random Forest: " + str(recall_rand_forest) + "\n")
        file.write("F1-score of Random Forest: " + str(f1_rand_forest) + "\n")
        file.write("RocAuc score of Random Forest: " + str(roc_auc_rand_forest) + "\n\n")

        file.write("Logistic Regression stats")
        file.write("Accuracy of Logistic Regression: " + str(acc_log_reg) + "\n")
        file.write("Precision of Logistic Regression: " + str(precision_log_reg) + "\n")
        file.write("Recall of Logistic Regression: " + str(recall_log_reg) + "\n")
        file.write("F1-score of Logistic Regression: " + str(f1_log_reg) + "\n")
        file.write("RocAuc score of Logistic Regression: " + str(roc_auc_log_reg) + "\n\n")

        file.write("Neural Networks stats")
        file.write("Accuracy of Neural Networks: " + str(acc_nn) + "\n")
        file.write("Precision of Neural Networks: " + str(precision_nn) + "\n")
        file.write("Recall of Neural Networks: " + str(recall_nn) + "\n")
        file.write("F1-score of Neural Networks: " + str(f1_nn) + "\n")
        file.write("RocAuc score of Neural Networks: " + str(roc_auc_nn) + "\n\n")

        file.close()


# test_different_sliding_window()
test_different_sliding_window_partial()
