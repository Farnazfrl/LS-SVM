import numpy as np
from scipy.linalg import block_diag, solve
from scipy.stats import mode
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

__all__ = [
    "LSSVM",
    "MultiClassLSSVM",
]


# --------------------------------------------------------------------------- #
# Shared kernel helper
# --------------------------------------------------------------------------- #

def rbf_kernel_matrix(X1: np.ndarray, X2: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """RBF kernel matrix K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)."""
    sq_dist = (
        np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        + np.sum(X2 ** 2, axis=1)
        - 2 * X1 @ X2.T
    )
    return np.exp(-gamma * sq_dist)


# --------------------------------------------------------------------------- #
# Single-view models
# --------------------------------------------------------------------------- #

class LSSVM:
    """Binary LS-SVM, dual formulation, RBF kernel."""

    def __init__(self, gamma: float = 1, gamma_rbf: float = 1, kernel: str = "rbf", ro: float = 1):
        self.kernel = kernel
        self.gamma_rbf = gamma_rbf
        self.gamma = gamma
        self.ro = ro

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray | int = 0) -> "LSSVM":
        n_samples = X.shape[0]
        Y = y.reshape(-1, 1)
        self.X = X
        self.Y = Y

        if self.kernel == "rbf":
            K = rbf_kernel_matrix(X, X, self.gamma_rbf)

        Omega = (Y * Y.T * K) + (np.eye(n_samples) / (self.gamma + self.ro * weights))
        P = np.block([[0, Y.T], [Y, Omega]])
        q = np.vstack([0, np.ones((n_samples, 1))])

        solution = np.linalg.solve(P, q)
        self.b = solution[0, 0]
        self.alpha = solution[1:].flatten()
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.kernel == "rbf":
            K_test = rbf_kernel_matrix(X_test, self.X, self.gamma_rbf)
            self.K_test = K_test
            self.y_pred = np.sum((K_test * self.alpha) * self.Y.T, axis=1) + self.b
        return np.sign(self.y_pred)


class MultiClassLSSVM:
    """One-vs-all multiclass LS-SVM from raw features (RBF kernel)."""

    def __init__(self, gamma: float = 1.0, gamma_rbf: float = 1.0, ro: float = 1.0):
        self.gamma = gamma
        self.gamma_rbf = gamma_rbf
        self.ro = ro

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray | int = 0) -> "MultiClassLSSVM":
        self.X_train = X
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        self.n_samples = n_samples
        self.n_classes = n_classes

        K = rbf_kernel_matrix(X, X, self.gamma_rbf)

        self.alphas_, self.biases_, self.y_binaries_ = [], [], []

        for idx, cls in enumerate(self.classes_):
            y_binary = np.where(y == cls, 1.0, -1.0)
            self.y_binaries_.append(y_binary)

            if isinstance(weights, int):
                D = np.eye(n_samples) / self.gamma
            else:
                D = np.diag(1.0 / (self.gamma + self.ro * weights[:, idx]))

            Y_outer = y_binary.reshape(-1, 1) @ y_binary.reshape(1, -1)
            Omega = Y_outer * K
            K_reg = Omega + D

            A = np.zeros((n_samples + 1, n_samples + 1))
            A[0, 1:] = y_binary
            A[1:, 0] = y_binary
            A[1:, 1:] = K_reg
            A += 1e-9 * np.eye(A.shape[0])

            b = np.zeros(n_samples + 1)
            b[1:] = 1.0

            solution = np.linalg.solve(A, b)
            self.biases_.append(solution[0])
            self.alphas_.append(solution[1:])

        self.biases_ = np.array(self.biases_)
        self.alphas_ = np.array(self.alphas_)
        self.y_binaries_ = np.array(self.y_binaries_)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        K_test = rbf_kernel_matrix(X_test, self.X_train, self.gamma_rbf)
        n_samples_test = X_test.shape[0]
        y_pred = np.zeros((n_samples_test, self.n_classes))
        for i in range(self.n_classes):
            y_pred[:, i] = (K_test @ (self.alphas_[i] * self.y_binaries_[i])) + self.biases_[i]
        self.y_pred = y_pred
        return np.argmax(y_pred, axis=1)
