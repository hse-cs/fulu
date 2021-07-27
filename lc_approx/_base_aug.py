from abc import ABC, abstractmethod

import numpy as np


def add_log_lam(passband, passband2lam):
    log_lam = np.array([passband2lam[i] for i in passband])
    return log_lam


def create_aug_data(t_min, t_max, passband_ids, n_obs=1000):
    t = []
    passband = []
    for band in passband_ids:
        t += list(np.linspace(t_min, t_max, n_obs))
        passband += [band] * n_obs
    return np.array(t), np.array(passband)


class BaseAugmentation(ABC):
    """
    Base abstract class for light curve augmentation

    Parameters:
    -----------
    passband2lam : dict
        A dictionary, where key is a passband ID and value is Log10 of its wave length.
        Example:
            passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23),
                             3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}
    """

    def __init__(self, passband2lam):
        self.passband2lam = passband2lam

    @abstractmethod
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
        raise NotImplemented

    @abstractmethod
    def predict(self, t, passband, copy=True):
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
        raise NotImplemented

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

        t_aug, passband_aug = create_aug_data(t_min, t_max, tuple(self.passband2lam), n_obs)
        flux_aug, flux_err_aug = self.predict(t_aug, passband_aug, copy=True)

        return t_aug, flux_aug, flux_err_aug, passband_aug
