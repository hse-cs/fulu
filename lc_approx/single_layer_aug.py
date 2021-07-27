import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def add_log_lam(passband, passband2lam):
    log_lam = np.array([passband2lam[i] for i in passband])
    return log_lam

def create_aug_data(t_min, t_max, n_passbands, n_obs=1000):
    t = []
    passbands = []
    for i_pb in range(n_passbands):
        t += list(np.linspace(t_min, t_max, n_obs))
        passbands += [i_pb] * n_obs
    return np.array(t), np.array(passbands)


class NNRegressor(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10):
        super(NNRegressor, self).__init__()
        self.seq = nn.Sequential(
                    nn.Linear(n_inputs, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, 1))
    def forward(self, x):
        return self.seq(x)
    
    
class FitNNRegressor(object):
    def __init__(self, n_hidden=10, n_epochs=10, batch_size=64, lr=0.01, lam=0., optimizer='Adam', debug=0):        
        self.model = None
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lam = lam
        self.optimizer = optimizer
        self.debug = debug
    
    def fit(self, X, y):
        # Estimate model
        self.model = NNRegressor(n_inputs=X.shape[1], n_hidden=self.n_hidden).to(device)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
        # Create dataset for trainig procedure
        train_data = TensorDataset(X_tensor, y_tensor)
        # Estimate loss
        loss_func = nn.MSELoss()
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Enable droout
        self.model.train(True)
        
        best_loss = float('inf')
        best_state = None
        
        # Start the model fit
        for epoch_i in range(self.n_epochs):
            loss_history = []
            for x_batch, y_batch in DataLoader(train_data, batch_size=self.batch_size, shuffle=True):
                # make prediction on a batch
                y_pred_batch = self.model(x_batch)
                loss = loss_func(y_batch, y_pred_batch)
                # zero the parameter gradients
                for param in self.model.parameters():
                    param.grad = None

                # backpropagate gradients
                loss.backward()
                # update the model weights
                opt.step()
                loss_history.append(loss.item())
            if self.debug:
                print("epoch: %i, mean loss: %.5f" % (epoch_i, np.mean(loss_history)))
            if np.mean(loss_history) <= best_loss:
                best_loss = np.mean(loss_history)
                best_state = self.model.state_dict()
        self.model.load_state_dict(best_state)
    
    def predict(self, X):
        with torch.no_grad():
            # Disable droout
            self.model.train(False)
            # Convert X and y into torch tensors
            X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
            # Make predictions for X 
            y_pred = self.model(X_tensor)
            y_pred = y_pred.cpu().detach().numpy()
            return y_pred
    
class SingleLayerNetAugmentation(object):
    
    def __init__(self, passband2lam):
        """
        Light Curve Augmentation based on NNRegressor with new features
        
        Parameters:
        -----------
        passband2lam : dict
            A dictionary, where key is a passband ID and value is Log10 of its wave length.
            Example: 
                passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23), 
                                 3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}
        """
       
        self.passband2lam = passband2lam

        self.ss_x = None
        self.ss_y = None
        self.ss_t = None
        self.reg = None

    def get_features(self, t, passband, ss_t):
        passband = np.array(passband)
        log_lam  = add_log_lam(passband, self.passband2lam)
        t        = ss_t.transform(np.array(t).reshape((-1, 1)))

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
        
        self.ss_t = StandardScaler().fit(np.array(t).reshape((-1, 1)))

        X = self.get_features(t, passband, self.ss_t)
        self.ss_x = StandardScaler().fit(X)
        X_ss = self.ss_x.transform(X)
        flux     = np.array(flux)
        
        self.ss_y = StandardScaler().fit(flux.reshape((-1, 1)))
        y_ss = self.ss_y.transform(flux.reshape((-1, 1)))
        self.reg = FitNNRegressor(n_hidden=80, n_epochs=100, batch_size=2, optimizer='SGD')
        self.reg.fit(X_ss, y_ss)
    
    
    def predict(self, t, passband, copy=True):
        """
        Apply the augmentation model to the given observation mjds.
        
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

        X = self.get_features(t, passband, self.ss_t)
        X_ss = self.ss_x.transform(X)
        
        flux_pred = self.ss_y.inverse_transform(self.reg.predict(X_ss))
        flux_err_pred = np.zeros(flux_pred.shape)

        return np.maximum(flux_pred, np.zeros(flux_pred.shape)), flux_err_pred
        
    
    def augmentation(self, t_min, t_max, n_obs=100):
        """
        The light curve augmentation.
        
        Parameters:
        -----------
        t_min, t_max : float
            Min and max timestamps of light curve observations.
        n_obs : int
            Number of observations in each passband required.
            
        Returns:
        --------
        t_aug : array-like
            Timestamps of light curve observations.
        flux_aug : array-like
            Flux of the light curve observations, approximated by the augmentation model.
        flux_err_aug : array-like
            Flux errors of the light curve observations, estimated by the augmentation model.
        passband_aug : array-like
            Passband IDs for each observation.
        """
        
        t_aug, passband_aug = create_aug_data(t_min, t_max, len(self.passband2lam), n_obs)
        flux_aug, flux_err_aug = self.predict(t_aug, passband_aug, copy=True)
        
        return t_aug, flux_aug, flux_err_aug, passband_aug