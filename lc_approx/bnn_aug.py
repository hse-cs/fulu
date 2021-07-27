import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


from lc_approx._base_aug import BaseAugmentation, create_aug_data


class BNNRegressor(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10):
        super(BNNRegressor, self).__init__()
        
        self.model = nn.Sequential(
                        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_inputs, out_features=n_hidden),
                        nn.LeakyReLU(),
                        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_hidden, out_features=n_hidden // 2),
                        nn.LeakyReLU(),
                        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_hidden // 2, out_features=1))
        
    def forward(self, x):
        return self.model(x)
    
    
class FitBNNRegressor:
    
    def __init__(self, n_hidden=10, n_epochs=10, lr=0.01, kl_weight=0.1, optimizer='Adam', debug=0, device='cpu'):
        self.model = None
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.lr = lr
        self.kl_weight = kl_weight
        self.optimizer = optimizer
        self.debug = debug
        self.device = torch.device(device)
        
    
    def fit(self, X, y):
        # Estimate model
        self.model = BNNRegressor(n_inputs=X.shape[1], n_hidden=self.n_hidden)
        self.model.to(self.device)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=self.device)
        # Estimate loss
        mse_loss = nn.MSELoss()
        kl_loss  = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        # Make predictions for X
        y_pred = self.model(X_tensor)
        y_pred = y_pred.cpu().detach().numpy()
        return y_pred
    
    def predict_n_times(self, X, ss, n_times=100):
        predictions = []
        for i in range(n_times):
            y_pred = self.predict(X)
            predictions.append(y_pred)
        predictions = ss.inverse_transform(np.array(predictions))
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return mean, std


class BayesianNetAugmentation(object):

    def __init__(self, passband2lam):
        """
        Light Curve Augmentation based on BNNRegressor with new features
        
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
        self.reg = None
    
    def get_features(self, t, passband):
        t        = np.array(t).reshape((-1, 1))
        passband = np.array(passband)
        log_lam  = add_log_lam(passband, self.passband2lam).reshape((-1, 1))
 
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

        X = self.get_features(t, passband)
        self.ss_x = StandardScaler().fit(X)
        X_ss = self.ss_x.transform(X)
        flux = np.array(flux)
        
        self.ss_y = StandardScaler().fit(flux.reshape((-1, 1)))
        y_ss = self.ss_y.transform(flux.reshape((-1, 1)))
        self.reg = FitBNNRegressor(n_hidden=40, n_epochs=400, lr=0.05, kl_weight=0.01, optimizer='Adam')
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

        X = self.get_features(t, passband)
        X_ss = self.ss_x.transform(X)
        
        flux_pred, flux_err_pred = self.reg.predict_n_times(X_ss, self.ss_y)

        return np.maximum(np.zeros(flux_pred.shape), flux_pred), flux_err_pred
        
    
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
        X_aug = self.ss_x.transform(self.get_features(t_aug, passband_aug))
        flux_aug, flux_err_aug = self.reg.predict_n_times(X_aug, self.ss_y)
        
        return t_aug, np.maximum(np.zeros(flux_aug.shape), flux_aug), flux_err_aug, passband_aug
