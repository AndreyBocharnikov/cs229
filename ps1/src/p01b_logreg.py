import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    predictions = util.probs_to_targets(clf.predict(x_eval))

    print("accuracy - ", (predictions == y_eval).sum() / y_eval.shape[0])
    util.plot(x_eval, y_eval, clf.theta, './../images/logres.png')
    np.savetxt(pred_path, predictions)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n, dtype=np.double)
        n_steps = 0
        while True:
            hessian = np.zeros((n, n))
            g = util.sigmoid(x, self.theta)
            for i in range(n):
                for j in range(n):
                    hessian[i][j] = np.sum(x[:, i] * x[:, j] * g * (1 - g))
            hessian /= m
            hessian_inv = np.linalg.inv(hessian)
            grad = -1 / m * np.sum(x * (y - g)[:, None], axis=0)
            delta = hessian_inv.dot(grad)
            if np.absolute(delta).sum() < self.eps:
                break
            self.theta -= self.step_size * delta
            n_steps += 1
        #print("converged in ", n_steps, " steps")
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        return util.sigmoid(x, self.theta)
        # *** END CODE HERE ***


if __name__ == '__main__':
    main('./../data/ds1_train.csv', './../data/ds1_valid.csv', './../data/ds1_pred.txt')
