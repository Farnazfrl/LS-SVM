import numpy as np
from scipy.linalg import block_diag, solve
from scipy.stats import mode
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

__all__ = [
    "LSSVM",
    "MultiClassLSSVM",
    "MultiViewLSSVM"
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


# --------------------------------------------------------------------------- #
# Coupled multi-view LS-SVM (Houthuys et al. 2018)
# --------------------------------------------------------------------------- #

class MultiViewLSSVM:
    """Multi-view LS-SVM with pairwise error coupling across views (raw features)."""

    def __init__(self, gamma, gamma_rbf: float = 1.0, ro: float = 10, ro_adaptive: float = 1):
        self.gamma = gamma          # per-view error regularization
        self.gamma_rbf = gamma_rbf  # RBF kernel width
        self.ro = ro                # coupling regularization
        self.ro_adaptive = ro_adaptive

    def get_params(self, deep: bool = True) -> dict:
        return {"gamma": self.gamma, "gamma_rbf": self.gamma_rbf, "ro": self.ro}

    def set_params(self, **params) -> "MultiViewLSSVM":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y, weights=0) -> None:
        self.X_train = X
        self.classes = np.unique(y)
        self.n_classes = int(len(self.classes))
        self.n_samples = int(X[0].shape[0])
        self.n_views = len(self.X_train)

        self.K = [rbf_kernel_matrix(X[v], X[v], self.gamma_rbf) for v in range(self.n_views)]

        if self.n_classes > 2:
            Y = np.zeros((self.n_samples, self.n_classes))
            for i, cls in enumerate(self.classes):
                Y[:, i] = np.where(y == cls, 1, -1)
            self.Y = Y
        else:
            Y = y
            self.Y = y

        block_gamma = np.array([])
        for v in range(self.n_views):
            diag_gamma = np.diag(np.array([self.gamma[v] + self.ro_adaptive * weights] * self.n_samples))
            block_gamma = block_diag(block_gamma, diag_gamma)
        block_gamma = block_gamma[1:, :]
        self.block_gamma = block_gamma

        I = np.zeros((self.n_samples * self.n_views, self.n_samples * self.n_views))
        for v1 in range(self.n_views):
            for v2 in range(self.n_views):
                if v1 != v2:
                    sl1 = slice(v1 * self.n_samples, (v1 + 1) * self.n_samples)
                    sl2 = slice(v2 * self.n_samples, (v2 + 1) * self.n_samples)
                    I[sl1, sl2] = np.eye(self.n_samples)

        if self.n_classes == 2:
            self.biases_multiclass, self.alphas_multiclass = self._solve_subproblem(Y, block_gamma, I)
        else:
            biases_multiclass = np.empty((self.n_views, 1))
            alphas_multiclass = np.empty((self.n_samples * self.n_views, 1))
            for i in range(self.n_classes):
                b_i, a_i = self._solve_subproblem(Y[:, i], block_gamma, I)
                biases_multiclass = np.hstack((biases_multiclass, b_i))
                alphas_multiclass = np.hstack((alphas_multiclass, a_i))
            self.biases_multiclass = biases_multiclass[:, 1:]
            self.alphas_multiclass = alphas_multiclass[:, 1:]

    def _solve_subproblem(self, y_binary, block_gamma, I):
        block_Y = block_diag(*[y_binary.reshape(-1, 1) for _ in range(self.n_views)])
        omegas = [y_binary.reshape(-1, 1) * y_binary.reshape(-1, 1).T * self.K[v] for v in range(self.n_views)]
        block_omega = block_diag(*omegas)

        p_bottom_left = (block_gamma @ block_Y) + (self.ro * I @ block_Y)
        p_bottom_right = (block_gamma @ block_omega) + np.eye(self.n_samples * self.n_views) + (self.ro * I @ block_omega)
        p = np.block([[np.zeros((self.n_views, self.n_views)), block_Y.T], [p_bottom_left, p_bottom_right]])
        p += 1e-4 * np.eye(p.shape[0])

        q_bottom = (block_gamma @ np.ones((self.n_samples * self.n_views, 1))) + \
                   ((self.n_views - 1) * self.ro * np.ones((self.n_samples * self.n_views, 1)))
        q = np.vstack([np.zeros((self.n_views, 1)), q_bottom])

        solution = solve(p, q)
        return solution[:self.n_views], solution[self.n_views:]

    def predict(self, X_test):
        K_test = [rbf_kernel_matrix(X_test[v], self.X_train[v], self.gamma_rbf) for v in range(self.n_views)]
        self.K_test = K_test
        n_samples_test = X_test[0].shape[0]
        self.n_samples_test = n_samples_test
        block_K_test = block_diag(*K_test)
        self.block_K_test = block_K_test

        if self.n_classes == 2:
            return self._predict_binary(self.Y, self.alphas_multiclass, self.biases_multiclass, n_samples_test)

        y_pred_multiclass = np.empty((n_samples_test, 1))
        for cls in range(self.n_classes):
            if self.n_views > 1:
                alphas_cls = self.alphas_multiclass[:, cls]
                biases_cls = self.biases_multiclass[:, cls]
            else:
                alphas_cls = self.alphas_multiclass[:, cls]
                biases_cls = self.biases_multiclass.flatten()[cls]
            y_pred = self._aggregate_views(self.Y[:, cls], alphas_cls, biases_cls, n_samples_test)
            y_pred_multiclass = np.hstack((y_pred_multiclass, y_pred.reshape(-1, 1)))

        self.y_pred_multiclass = y_pred_multiclass[:, 1:]
        return np.argmax(self.y_pred_multiclass, axis=1)

    def _predict_binary(self, y_binary, alphas, biases, n_samples_test):
        if self.n_views > 1:
            block_alphas = block_diag(*[alphas[i * self.n_samples:(i + 1) * self.n_samples].reshape(-1, 1) for i in range(self.n_views)])
            block_biases = block_diag(*[np.array([biases[i]] * n_samples_test).reshape(-1, 1) for i in range(self.n_views)])
        else:
            block_alphas = alphas.reshape(-1, 1)
            block_biases = np.array([biases.flatten()] * n_samples_test).reshape(-1, 1)

        block_Y = block_diag(*[y_binary.reshape(-1, 1) for _ in range(self.n_views)])
        y_pred_block = self.block_K_test @ (block_alphas * block_Y) + block_biases

        y_pred = np.zeros((n_samples_test, self.n_views))
        j = 0
        for i in range(self.n_views):
            y_pred[:, i] = y_pred_block[j:j + n_samples_test, i]
            j += n_samples_test

        self.y_pred_multiclass = np.mean(y_pred, axis=1)
        return np.sign(self.y_pred_multiclass)

    def _aggregate_views(self, y_binary, alphas_cls, biases_cls, n_samples_test):
        if self.n_views > 1:
            block_alphas = block_diag(*[alphas_cls[i * self.n_samples:(i + 1) * self.n_samples].reshape(-1, 1) for i in range(self.n_views)])
            block_biases = block_diag(*[np.array([biases_cls[i]] * n_samples_test).reshape(-1, 1) for i in range(self.n_views)])
        else:
            block_alphas = alphas_cls.reshape(-1, 1)
            block_biases = np.array([biases_cls] * n_samples_test).reshape(-1, 1)

        block_Y = block_diag(*[y_binary.reshape(-1, 1) for _ in range(self.n_views)])
        y_pred_block = self.block_K_test @ (block_alphas * block_Y) + block_biases

        y_pred = np.zeros((n_samples_test, self.n_views))
        j = 0
        for i in range(self.n_views):
            y_pred[:, i] = y_pred_block[j:j + n_samples_test, i]
            j += n_samples_test
        return np.mean(y_pred, axis=1)
