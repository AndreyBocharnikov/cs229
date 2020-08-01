import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'

def get_data(train_path, valid_path, test_path, label_col):
    x_train, y_train = util.load_dataset(train_path, label_col=label_col, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, label_col=label_col, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col=label_col, add_intercept=True)
    return (x_train, y_train, x_eval, y_eval, x_test, y_test)


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, y_train_true, x_eval, y_eval_true, x_test, y_test_true = get_data(train_path, valid_path, test_path, 't')
    clf = LogisticRegression()
    clf.fit(x_train, y_train_true)
    predicted_targets = util.probs_to_targets(clf.predict(x_test))

    np.savetxt(pred_path_c, predicted_targets)
    util.plot(x_test, y_test_true, clf.theta, './../images/posonly_c')
    print((predicted_targets == y_test_true).mean())

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    _, y_train, _, y_eval, _, y_test = get_data(train_path, valid_path, test_path, 'y')
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    predicted_targets = util.probs_to_targets(predictions)

    np.savetxt(pred_path_d, predicted_targets)
    util.plot(x_test, y_test_true, clf.theta, './../images/posonly_d')
    print((predicted_targets == y_test_true).mean())

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_eval_pos = x_eval[y_eval_true == 1]
    alpha = clf.predict(x_eval_pos).mean()
    predictions /= alpha
    predicted_targets = util.probs_to_targets(predictions)

    np.savetxt(pred_path_e, predicted_targets)
    util.plot(x_test, y_test_true, clf.theta, './../images/posonly_c')
    print((predicted_targets == y_test_true).mean())

    # *** END CODER HERE

if __name__ == '__main__':
    main('./../data/ds3_train.csv', './../data/ds3_valid.csv', './../data/ds3_test.csv', './../data/ds3_pred_X.txt')
