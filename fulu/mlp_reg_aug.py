import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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
    """

    def __init__(self, passband2lam):
        super().__init__(passband2lam)

        self.ss_x = None
        self.ss_y = None
        self.ss_t = None
        self.reg = None
    
    def _preproc_features(self, t, passband, ss_t):
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
            Timestamps of light curve observations.
        flux : array-like
            Flux of the light curve observations.
        flux_err : array-like
            Flux errors of the light curve observations.
        passband : array-like
            Passband IDs for each observation.
        """
        
        self.ss_t = StandardScaler().fit(np.array(t).reshape((-1, 1)))

        X = self._preproc_features(t, passband, self.ss_t)
        self.ss_x = StandardScaler().fit(X)
        X_ss = self.ss_x.transform(X)
        flux     = np.array(flux)
        
        self.ss_y = StandardScaler().fit(flux.reshape((-1, 1)))
        y_ss = self.ss_y.transform(flux.reshape((-1, 1)))

        self.reg = MLPRegressor(hidden_layer_sizes=(20,10,), solver='lbfgs', activation='tanh',
                                learning_rate_init=0.001, max_iter=90, batch_size=1)
        self.reg.fit(X_ss, y_ss.reshape(-1))
        return self

    def predict(self, t, passband):
        """
        Apply the augmentation model to the given observation mjds.
        
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
        
        flux_pred = self.ss_y.inverse_transform(self.reg.predict(X_ss))
        flux_err_pred = np.zeros(flux_pred.shape)

        return np.maximum(flux_pred, np.zeros(flux_pred.shape)), flux_err_pred
