import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C

from fulu._base_aug import BaseAugmentation, add_log_lam


class GaussianProcessesAugmentation(BaseAugmentation):
    """
    Light Curve Augmentation based on Gaussian Processes Regression

    Parameters:
    -----------
    passband2lam : dict
        A dictionary, where key is a passband ID and value is Log10 of its wave length.
        Example:
            passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23),
                             3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}
    kernel : kernel object from sklearn
        Kernel for GaussianProcessRegressor. Can be combine from different kernels.
        Example:
            kernel = C(1.0)*RBF([1, 1]) + Matern() + WhiteKernel()
    use_err : bool
        Flag responsible for using flux error by GP Regressor.
    """

    def __init__(self, passband2lam, kernel=C(1.0) * RBF([1.0, 1.0]) + WhiteKernel(), use_err=False):

        super().__init__(passband2lam)

        self.ss = None
        self.reg = None
        self.kernel = kernel
        self.use_err = use_err

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

        log_lam = add_log_lam(passband, self.passband2lam)

        t = np.array(t)
        flux = np.array(flux)
        flux_err = np.array(flux_err)
        passband = np.array(passband)
        log_lam = add_log_lam(passband, self.passband2lam)

        X = np.concatenate((t.reshape(-1, 1), log_lam.reshape(-1, 1)), axis=1)

        X_error = (flux_err / np.std(flux)).reshape(-1)
        self.ss = StandardScaler()
        X_ss = self.ss.fit_transform(X)

        reg_kwargs = dict(
            kernel=self.kernel, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=5, normalize_y=True, random_state=42
        )

        if self.use_err:
            reg_kwargs["alpha"] = X_error**2

        self.reg = GaussianProcessRegressor(**reg_kwargs)

        self.reg.fit(X_ss, flux)
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

        t = np.array(t)
        passband = np.array(passband)
        log_lam = add_log_lam(passband, self.passband2lam)

        X = np.concatenate((t.reshape(-1, 1), log_lam.reshape(-1, 1)), axis=1)
        X_ss = self.ss.transform(X)

        flux_pred, flux_err_pred = self.reg.predict(X_ss, return_std=True)

        return flux_pred, flux_err_pred
