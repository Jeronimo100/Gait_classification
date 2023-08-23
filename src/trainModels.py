import pickle


def run_svm(x_train, x_test, y_train, sequence_label, sliding_window, partial):
    from sklearn.svm import SVC

    x_test_sequence_label = sequence_label.astype(int)

    svm = SVC(random_state=1, cache_size=4000)
    try:
        svm = pickle.load(open('../models/svm_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial), 'rb'))
        print("svm has been retrieved from saved model")
    except FileNotFoundError:
        print("training svm model...")
        svm.fit(x_train, y_train)
        try:
            pickle.dump(svm, open('../models/svm_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial), 'wb'))
            print("svm has been trained and saved.")
        except FileNotFoundError:
            print("was not able to save the model")
            print("svm has been trained but not saved.")

    y_pred = svm.predict(x_test)

    y_pred = get_aggregate_sequence_prediction(y_pred, x_test_sequence_label)

    return y_pred


def run_bayes(x_train, x_test, y_train, sequence_label, sliding_window, partial):
    from sklearn.naive_bayes import GaussianNB

    x_test_sequence_label = sequence_label.astype(int)

    nb = GaussianNB()
    # try to retrieve model for that sliding window value, compute otherwise
    try:
        nb = pickle.load(open('../models/bayes_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial), 'rb'))
        print("bayes has been retrieved from saved model")
    except FileNotFoundError:
        print("training bayes model...")
        nb.fit(x_train, y_train)
        try:
            pickle.dump(nb, open('../models/bayes_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial), 'wb'))
            print("bayes has been trained and saved.")
        except FileNotFoundError:
            print("was not able to save the model")
            print("bayes has been trained but not saved.")

    # accuracy measured on subsequences without taking into account the overall sequence
    # print('Accuracy of dec tree in test data is:', nb.score(x_test, y_test))

    y_pred = nb.predict(x_test)

    y_pred = get_aggregate_sequence_prediction(y_pred, x_test_sequence_label)

    return y_pred


def run_dec_tree(x_train, x_test, y_train, sequence_label, sliding_window, partial):
    from sklearn.tree import DecisionTreeClassifier

    x_test_sequence_label = sequence_label.astype(int)
    dt = DecisionTreeClassifier()

    # try to retrieve model for that sliding window value, compute otherwise
    try:
        dt = pickle.load(open('../models/dec_tree_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial),
                              'rb'))
        print("decision tree has been retrieved from saved model")
    except FileNotFoundError:
        print("training decision tree model...")
        dt.fit(x_train, y_train)
        try:
            pickle.dump(dt, open('../models/dec_tree_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial),
                                 'wb'))
            print("decision tree has been trained and saved.")
        except FileNotFoundError:
            print("was not able to save the model")
            print("decision tree has been trained but not saved.")

    # accuracy measured on subsequences without taking into account the overall sequence
    # print('Accuracy of dec tree in test data is:', dt.score(x_test, y_test))

    y_pred = dt.predict(x_test)

    y_pred = get_aggregate_sequence_prediction(y_pred, x_test_sequence_label)

    return y_pred


def run_random_forest(x_train, x_test, y_train, sequence_label, sliding_window, partial):
    from sklearn.ensemble import RandomForestClassifier

    x_test_sequence_label = sequence_label.astype(int)
    rf = RandomForestClassifier()

    # try to retrieve model for that sliding window value, compute otherwise
    try:
        rf = pickle.load(open('../models/rand_for_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial),
                              'rb'))
        print("random forest has been retrieved from saved model")
    except FileNotFoundError:
        print("training random forest model...")
        rf.fit(x_train, y_train)
        try:
            pickle.dump(rf, open('../models/rand_for_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial),
                                 'wb'))
            print("random forest has been trained and saved.")
        except FileNotFoundError:
            print("was not able to save the model")
            print("random forest has been trained but not saved.")

    # accuracy measured on subsequences without taking into account the overall sequence
    # print('Accuracy of random forest in test data is:', rf.score(x_test, y_test))

    y_pred = rf.predict(x_test)

    y_pred = get_aggregate_sequence_prediction(y_pred, x_test_sequence_label)

    return y_pred


def run_log_reg(x_train, x_test, y_train, sequence_label, sliding_window, partial):
    from sklearn.linear_model import LogisticRegression

    x_test_sequence_label = sequence_label.astype(int)

    lr = LogisticRegression(random_state=0, max_iter=100000)

    # try to retrieve model for that sliding window value, compute otherwise
    try:
        lr = pickle.load(open('../models/log_reg_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial),
                              'rb'))
        print("logistic regression has been retrieved from saved model")
    except FileNotFoundError:
        print("training logistic regression model...")
        lr.fit(x_train, y_train)
        try:
            pickle.dump(lr, open('../models/log_reg_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial),
                                 'wb'))
            print("logistic regression has been trained and saved.")
        except FileNotFoundError:
            print("was not able to save the model")
            print("logistic regression has been trained but not saved.")

    # accuracy measured on subsequences without taking into account the overall sequence
    # print('Accuracy of logistic regression in test data is:', lr.score(x_test, y_test))

    y_pred = lr.predict(x_test)

    y_pred = get_aggregate_sequence_prediction(y_pred, x_test_sequence_label)

    return y_pred


# multi-layer perceptron
def run_mlp_classifier(x_train, x_test, y_train, sequence_label, sliding_window, partial):
    from sklearn.neural_network import MLPClassifier

    x_test_sequence_label = sequence_label.astype(int)

    mlp = MLPClassifier(max_iter=10000, random_state=1, activation='logistic')

    # try to retrieve model for that sliding window value, compute otherwise
    try:
        mlp = pickle.load(open('../models/mlp_class_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial),
                               'rb'))
        print("neural network has been retrieved from saved model")
    except FileNotFoundError:
        print("training neural network model...")
        mlp.fit(x_train, y_train)
        try:
            pickle.dump(mlp, open('../models/mlp_class_{sw}{partial}.sav'.format(sw=sliding_window, partial=partial),
                                  'wb'))
            print("neural network has been trained and saved.")
        except FileNotFoundError:
            print("was not able to save the model")
            print("neural network has been trained but not saved.")

    # accuracy measured on subsequences without taking into account the overall sequence
    # print('Accuracy of neural network in test data is:', mlp.score(x_test, y_test))

    y_pred = mlp.predict(x_test)

    y_pred = get_aggregate_sequence_prediction(y_pred, x_test_sequence_label)

    return y_pred


def get_aggregate_sequence_prediction(prediction, x_test_sequence_label):
    temp_result = prediction
    majority_votes = [0] * 81

    # compute majority vote
    for i in range(0, len(temp_result)):
        if temp_result[i] == 1.0:
            majority_votes[x_test_sequence_label[i]] += 1
        else:
            majority_votes[x_test_sequence_label[i]] -= 1

    # change votes according to majority
    for i in range(0, len(temp_result)):
        if majority_votes[x_test_sequence_label[i]] <= 0:
            temp_result[i] = 0
        else:
            temp_result[i] = 1

    return temp_result
