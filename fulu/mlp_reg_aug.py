import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from fulu._base_aug import BaseAugmentation, add_log_lam


class MLPRegressionAugmentation(BaseAugmentation):
    """
    Light Curve Augmentation based on scikit-learn MLPRegressor

    Parameters:
    -----------
    passband2lam : dict
        A dictionary, where key is a passband ID and value is Log10 of its wave length.
        Example:
            passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23),
                             3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}
    hidden_layer_sizes : tuple
        The ith element represents the number of neurons in the ith hidden layer, length = n_layers - 2.
    solver : string
        The solver for weight optimization.
    activation : string
        Activation function for the hidden layer. Possible values: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}.
    learning_rate_init : float
        The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’
        or ‘adam’.
    max_iter : int
        Maximum number of iterations. The solver iterates until convergence or this number of iterations.
        For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs
        (how many times each data point will be used), not the number of gradient steps.
    batch_size : int
        Size of minibatches for stochastic optimizers. If the solver is ‘lbfgs’,
        the classifier will not use minibatch. When set to “auto”, batch_size=min(200, n_samples)
    weight_decay : float
        L2 penalty (regularization term) parameter.
    """

    def __init__(
        self,
        passband2lam,
        hidden_layer_sizes=(
            20,
            10,
        ),
        solver="lbfgs",
        activation="tanh",
        learning_rate_init=0.001,
        max_iter=90,
        batch_size=1,
        weight_decay=0.0001,
    ):
        super().__init__(passband2lam)

        self.ss_x = None
        self.ss_y = None
        self.ss_t = None
        self.reg = None
        self.flux_err = 0

        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.weight_decay = weight_decay

    def _preproc_features(self, t, passband, ss_t):
        passband = np.array(passband)
        log_lam = add_log_lam(passband, self.passband2lam)
        t = ss_t.transform(np.array(t).reshape((-1, 1)))

        X = np.concatenate((t, log_lam.reshape((-1, 1))), axis=1)
        return X

    def fit(self, t, flux, flux_err, passband):
        """
        Fit an augmentation model.

        Parameters:
        -----------
        t : array-like
            Timestamps of light curve observations.
        flux : array-like
            Flux of the light curve observations.
        flux_err : array-like
            Flux errors of the light curve observations.
        passband : array-like
            Passband IDs for each observation.
        """

        super().fit(t, flux, flux_err, passband)

        self.ss_t = StandardScaler().fit(np.array(t).reshape((-1, 1)))

        X = self._preproc_features(t, passband, self.ss_t)
        self.ss_x = StandardScaler().fit(X)
        X_ss = self.ss_x.transform(X)
        flux = np.array(flux)

        self.ss_y = StandardScaler().fit(flux.reshape((-1, 1)))
        y_ss = self.ss_y.transform(flux.reshape((-1, 1)))

        self.reg = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            solver=self.solver,
            activation=self.activation,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            alpha=self.weight_decay,
        )
        self.reg.fit(X_ss, y_ss)

        flux_pred = self.ss_y.inverse_transform(self.reg.predict(X_ss).reshape(-1, 1)).reshape(-1)
        self.flux_err = np.sqrt(mean_squared_error(flux, flux_pred))
        return self

    def predict(self, t, passband):
        """
        Apply the augmentation model to the given observation time moments.

        Parameters:
        -----------
        t : array-like
            Timestamps of light curve observations.
        passband : array-like
            Passband IDs for each observation.

        Returns:
        --------
        flux_pred : array-like
            Flux of the light curve observations, approximated by the augmentation model.
        flux_err_pred : array-like
            Flux errors of the light curve observations, estimated by the augmentation model.
        """

        X = self._preproc_features(t, passband, self.ss_t)
        X_ss = self.ss_x.transform(X)

        flux_pred = self.ss_y.inverse_transform(self.reg.predict(X_ss).reshape(-1, 1)).reshape(-1)
        flux_err_pred = np.full_like(flux_pred, self.flux_err)

        return np.maximum(flux_pred, np.zeros(flux_pred.shape)), flux_err_pred
