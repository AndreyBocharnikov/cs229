import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    best_clf, best_score, best_tau = None, None, None
    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_eval)
        mse = np.sqrt(((predictions - y_eval) ** 2).mean())
        plt.plot(x_train, y_train, "bx", x_eval, predictions, "ro")
        plt.show()
        if best_score == None or mse < best_score:
            best_score = mse
            best_clf = clf
            best_tau = tau
        # print("tau = ", tau, " mse equels to ", mse)

    print("The best tau on val set is", best_tau, "mse equels to ", best_score)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    predictions = best_clf.predict(x_test)
    mse = np.sqrt(((predictions - y_test) ** 2).mean())
    print("mse on test dataset with best tau equels to", mse)

    np.savetxt(pred_path, predictions)
    # *** END CODE HERE ***

if __name__ == '__main__':
    main([0.005, 0.01, 0.05, 0.1, 0.25, 0.5], './../data/ds5_train.csv', './../data/ds5_valid.csv', './../data/ds5_test.csv', './../data/ds5_pred.txt')
