import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from fulu._base_aug import BaseAugmentation, add_log_lam


class BNNRegressor(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10, prior_sigma=0.01):
        super(BNNRegressor, self).__init__()

        self.model = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma, in_features=n_inputs, out_features=n_hidden),
            nn.Tanh(),
            # bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma, in_features=n_hidden, out_features=n_hidden),
            # nn.Tanh(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma, in_features=n_hidden, out_features=1),
        )

    def forward(self, x):
        return self.model(x)


class FitBNNRegressor:
    def __init__(
        self,
        n_hidden=10,
        prior_sigma=0.01,
        n_epochs=10,
        lr=0.01,
        kl_weight=0.1,
        optimizer="Adam",
        debug=0,
        device="cpu",
        weight_decay=0,
    ):
        self.model = None
        self.n_hidden = n_hidden
        self.prior_sigma = prior_sigma
        self.n_epochs = n_epochs
        self.lr = lr
        self.kl_weight = kl_weight
        self.optimizer = optimizer
        self.debug = debug
        self.device = torch.device(device)
        self.weight_decay = weight_decay

    def fit(self, X, y):
        # Estimate model
        self.model = BNNRegressor(n_inputs=X.shape[1], n_hidden=self.n_hidden, prior_sigma=self.prior_sigma)
        self.model.to(self.device)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=self.device)
        # Estimate loss
        mse_loss = nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Enable droout
        self.model.train(True)
        # Start the model fit
        for epoch_i in range(self.n_epochs):
            y_pred = self.model(X_tensor)
            mse = mse_loss(y_pred, y_tensor)
            kl = kl_loss(self.model)
            loss = mse + self.kl_weight * kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(self, X, n_times=1):
        # Disable droout
        self.model.train(False)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        # Make predictions for X
        y_pred = self.model(X_tensor)
        y_pred = y_pred.cpu().detach().numpy()
        return y_pred

    def predict_n_times(self, X, ss, n_times=100):
        predictions = []
        for i in range(n_times):
            y_pred = self.predict(X).reshape(
                -1,
            )
            predictions.append(y_pred)
        predictions = ss.inverse_transform(np.array(predictions))
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return mean, std


class BayesianNetAugmentation(BaseAugmentation):
    """
    Light Curve Augmentation based on BNNRegressor with new features

    Parameters:
    -----------
    passband2lam : dict
        A dictionary, where key is a passband ID and value is Log10 of its wave length.
        Example:
            passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23),
                             3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}
    n_hidden : int
        A number of hidden neurons in the network.
    prior_sigma : float
        The standard deviation for a neuron weights prior distribution.
    n_epochs : int
        The number of epochs in training.
    lr : float
        The learning rate for the network optimization.
    kl_weight : float
        The coefficient for a regularization term of a loss function.
    optimizer : string
        Optimization algorithm name. Possible values: 'Adam', 'RMSprop', 'SGD'.
    device : str or torch device
        Torch device name, default is 'cpu'
    weight_decay : float
        L2 penalty (regularization term) parameter.
    """

    def __init__(
        self,
        passband2lam,
        n_hidden=20,
        prior_sigma=0.1,
        n_epochs=3000,
        lr=0.01,
        kl_weight=0.0001,
        optimizer="Adam",
        device="cpu",
        weight_decay=0,
    ):
        super().__init__(passband2lam)

        self.n_hidden = n_hidden
        self.prior_sigma = prior_sigma
        self.n_epochs = n_epochs
        self.lr = lr
        self.kl_weight = kl_weight
        self.optimizer = optimizer
        self.device = device
        self.weight_decay = weight_decay

        self.ss_x = None
        self.ss_y = None
        self.reg = None

    def _preproc_features(self, t, passband):
        t = np.array(t).reshape((-1, 1))
        passband = np.array(passband)
        log_lam = add_log_lam(passband, self.passband2lam).reshape((-1, 1))

        X = np.concatenate((t, log_lam), axis=1)
        return X

    def fit(self, t, flux, flux_err, passband):
        """
        Fit an augmentation model.

        Parameters:
        -----------
        t : array-like
            Scaled timestamps of light curve observations.
        flux : array-like
            Flux of the light curve observations.
        flux_err : array-like
            Flux errors of the light curve observations.
        passband : array-like
            Passband IDs for each observation.
        """
        super().fit(t, flux, flux_err, passband)

        X = self._preproc_features(t, passband)
        self.ss_x = StandardScaler().fit(X)
        X_ss = self.ss_x.transform(X)
        flux = np.array(flux)

        self.ss_y = StandardScaler().fit(flux.reshape((-1, 1)))
        y_ss = self.ss_y.transform(flux.reshape((-1, 1)))
        self.reg = FitBNNRegressor(
            n_hidden=self.n_hidden,
            prior_sigma=self.prior_sigma,
            n_epochs=self.n_epochs,
            lr=self.lr,
            kl_weight=self.kl_weight,
            optimizer=self.optimizer,
            device=self.device,
            weight_decay=self.weight_decay,
        )
        self.reg.fit(X_ss, y_ss)
        return self

    def predict(self, t, passband):
        """
        Apply the augmentation model to the given observation time moments.

        Parameters:
        -----------
        t : array-like
            Scaled timestamps of light curve observations.
        passband : array-like
            Passband IDs for each observation.

        Returns:
        --------
        flux_pred : array-like
            Flux of the light curve observations, approximated by the augmentation model.
        flux_err_pred : array-like
            Flux errors of the light curve observations, estimated by the augmentation model.
        """

        X = self._preproc_features(t, passband)
        X_ss = self.ss_x.transform(X)

        flux_pred, flux_err_pred = self.reg.predict_n_times(X_ss, self.ss_y)

        return np.maximum(np.zeros(flux_pred.shape), flux_pred), flux_err_pred
