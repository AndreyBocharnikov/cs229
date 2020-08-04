import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    clf = PoissonRegression(step_size=lr, max_iter=10000)
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    predictions = clf.predict(x_eval)

    relative = np.absolute(1 - predictions / y_eval).mean()
    print("Relative error - ", relative)
    np.savetxt(pred_path, predictions)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        cnt = 0
        converged = False
        self.theta = np.zeros(n, dtype=np.double)
        for i in range(self.max_iter):
            h = np.exp(np.dot(x, self.theta))
            delta = np.mean((y - h)[:, None] * x, axis=0)
            if np.absolute(delta).sum() < self.eps:
                converged = True
                break
            self.theta += self.step_size * delta
            cnt += 1
        if not converged:
            print("Did not converge, increase maximum number of iterations.")
        else:
            print("Converged in ", cnt, " iterations")
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(np.dot(x, self.theta))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(0.0000001, './../data/ds4_train.csv', './../data/ds4_valid.csv', './../data/ds4_pred.txt')
