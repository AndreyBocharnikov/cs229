import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    predictions = clf.predict(x_eval)
    mse = np.sqrt(((predictions - y_eval) ** 2).mean())
    print("mse equels to ", mse)

    plt.title("Data")
    plt.xlabel('input')
    plt.ylabel('tagret')
    plt.plot(x_train, y_train, "bx", x_eval, predictions, "ro")
    plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        result = np.zeros((m), dtype=np.double)
        for i in range(m):
            w = np.exp(-((self.x - x[i][None, :]) ** 2).sum(axis=1) / (2 * self.tau ** 2))
            W = np.diag(w)
            self.theta = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y)
            result[i] = np.dot(x[i], self.theta)
        return result
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(0.5, './../data/ds5_train.csv', './../data/ds5_valid.csv')

