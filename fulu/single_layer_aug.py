import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from fulu._base_aug import BaseAugmentation, add_log_lam


class NNRegressor(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10, activation="tanh"):
        super(NNRegressor, self).__init__()

        act = nn.Tanh()
        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        else:
            raise ValueError("activation function {} is not supported".format(activation))

        self.seq = nn.Sequential(nn.Linear(n_inputs, n_hidden), act, nn.Linear(n_hidden, 1))

    def forward(self, x):
        return self.seq(x)


class FitNNRegressor:
    def __init__(
        self,
        n_hidden=10,
        activation="tanh",
        n_epochs=10,
        batch_size=64,
        lr=0.01,
        weight_decay=0.0,
        optimizer="Adam",
        debug=0,
        device="auto",
    ):
        self.model = None
        self.n_hidden = n_hidden
        self.activation = activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.debug = debug

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def fit(self, X, y):
        # Estimate model
        self.model = NNRegressor(n_inputs=X.shape[1], n_hidden=self.n_hidden, activation=self.activation).to(
            self.device
        )
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        # Create dataset for trainig procedure
        train_data = TensorDataset(X_tensor, y_tensor)
        # Estimate loss
        loss_func = nn.MSELoss()
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError('optimizer "{}" is not supported'.format(self.optimizer))
        # Enable dropout
        self.model.train(True)
        # Start the model fit
        for epoch_i in range(self.n_epochs):
            for x_batch, y_batch in DataLoader(train_data, batch_size=self.batch_size, shuffle=True):
                # make prediction on a batch
                y_pred_batch = self.model(x_batch)
                loss = loss_func(y_batch, y_pred_batch)
                # zero the parameter gradients
                opt.zero_grad()
                # backpropagate gradients
                loss.backward()
                # update the model weights
                opt.step()

    def predict(self, X):
        with torch.no_grad():
            # Disable droout
            self.model.train(False)
            # Convert X and y into torch tensors
            X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            # Make predictions for X
            y_pred = self.model(X_tensor)
            y_pred = y_pred.cpu().detach().numpy()
            return y_pred


class SingleLayerNetAugmentation(BaseAugmentation):
    """
    Light Curve Augmentation based on NNRegressor with new features

    Parameters:
    -----------
    passband2lam : dict
        A dictionary, where key is a passband ID and value is Log10 of its wave length.
        Example:
            passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23),
                             3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}
    n_hidden : int
        Number of neurons in a layer.
    activation : string
        Neuron's activation function. Possible values: {'tanh', 'relu', 'sigmoid'}.
    n_epochs : int
        Number of epochs of model weights optimization.
    batch_size : int
        Number of samples for one iteration of model weights optimization.
    lr : float
        Learning rate value for model weights optimization.
    optimizer : string
        Optimization algorithm. Possible values: {'SGD', 'Adam', 'RMSprop'}.
    device : str or torch device, optional
        Torch device name, default is 'auto' which uses CUDA if available and CPU if not
    weight_decay : float
        L2 penalty (regularization term) parameter.
    """

    def __init__(
        self,
        passband2lam,
        n_hidden=20,
        activation="tanh",
        n_epochs=1000,
        batch_size=500,
        lr=0.01,
        optimizer="Adam",
        device="auto",
        weight_decay=0,
    ):
        super().__init__(passband2lam)

        self.n_hidden = n_hidden
        self.activation = activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.device = device
        self.weight_decay = weight_decay

        self.ss_x = None
        self.ss_y = None
        self.ss_t = None
        self.reg = None
        self.flux_err = 0

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
            Scaled timestamps of light curve observations.
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
        self.reg = FitNNRegressor(
            n_hidden=self.n_hidden,
            activation=self.activation,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            optimizer=self.optimizer,
            device=self.device,
            weight_decay=self.weight_decay,
        )
        self.reg.fit(X_ss, y_ss)

        flux_pred = self.ss_y.inverse_transform(self.reg.predict(X_ss)).reshape(
            -1,
        )
        self.flux_err = np.sqrt(mean_squared_error(flux, flux_pred))
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

        X = self._preproc_features(t, passband, self.ss_t)
        X_ss = self.ss_x.transform(X)

        flux_pred = self.ss_y.inverse_transform(self.reg.predict(X_ss)).reshape(
            -1,
        )
        flux_err_pred = np.full_like(flux_pred, self.flux_err)

        return np.maximum(flux_pred, np.zeros(flux_pred.shape)), flux_err_pred
