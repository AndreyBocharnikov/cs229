import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    clf = GDA()
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    predictions = clf.predict(x_eval)
    util.plot(x_eval, y_eval, util.merge_thethas(clf.theta), './../plots/gda.png')
    print("accuracy", (predictions == y_eval).sum())
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        y = y.astype(int)
        m, n = x.shape
        ids = x[:, 1] <= 400 # task h, accuracy from 0.83 to 0.9
        x = x[ids]
        y = y[ids]
        positive_labels = y.sum()
        phi = positive_labels / m
        mu = [x[y == 0].sum(axis=0) / (m - positive_labels), x[y == 1].sum(axis=0) / positive_labels]
        sigma = 1 / m * np.sum(np.dot((x[i] - mu[y[i]])[:, None], (x[i] - mu[y[i]])[None, :]) for i in range(x.shape[0]))
        sigma_inv = np.linalg.inv(sigma)
        theta = [np.dot(sigma_inv[j], mu[1] - mu[0]) for j in range(n)]
        theta0 = -np.log((1 - phi) / phi) + 1 / 2 * np.sum(sigma_inv * (np.dot(mu[0][:, None], mu[0][None, :]) - np.dot(mu[1][:, None], mu[1][None, :])))
        self.theta = (theta, theta0)
        return (theta, theta0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = 1 / (1 + np.exp(-(np.dot(x, self.theta[0]) + self.theta[1])))
        return util.probs_to_targets(z)
        # *** END CODE HERE

if __name__ == '__main__':
    main('./../data/ds1_train.csv', './../data/ds1_valid.csv', './../data/ds1_gda_pred.txt')
