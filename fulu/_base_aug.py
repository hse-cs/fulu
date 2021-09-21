from abc import ABC, abstractmethod
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import fulu.plotting
from fulu.plotting import Plotting_lc


def add_log_lam(passband, passband2lam):
    """
    """
    
    log_lam = np.array([passband2lam[i] for i in passband])
    return log_lam

def create_aug_data(t_min, t_max, passband_ids, n_obs=1000):
    """
    """
    
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
        self.plot = fulu.plotting.Plotting_lc(passband2lam)
        self.t_train = None
        self.flux_train = None
        self.flux_err_train = None
        self.passband_train = None
        
        self.t_approx = None
        self.flux_approx = None
        self.flux_err_approx = None
        self.passband_approx = None

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
        flux_aug, flux_err_aug = self.predict(t_aug, passband_aug)
        
        self.t_approx = t_aug
        self.flux_approx = flux_aug
        self.flux_err_approx = flux_err_aug
        self.passband_approx = passband_aug

        return t_aug, flux_aug, flux_err_aug, passband_aug
    
                    
    def _plot_one_graph_passband(self, t_train, flux_train, flux_err_train, passband_train, passband, ax, plot_approx, plot_peak, n_obs):
        """
        """
        
        Plotting_lc(self.passband2lam).errorbar_passband(t_train, flux_train, flux_err_train, passband_train, passband, ax)
        
        
        if plot_approx:
            anobject_approx = self.plot._make_dataframe(self.t_approx, self.flux_approx, self.flux_err_approx, self.passband_approx)
            anobject_approx = anobject_approx.sort_values('time')
            light_curve_approx = anobject_approx[anobject_approx.passband == passband]
            ax.plot(light_curve_approx['time'].values, light_curve_approx['flux'].values,
                        linewidth=3.5, color=self.plot.colors[passband], label=str(passband) + ' approx flux', zorder=10)
            ax.fill_between(light_curve_approx['time'].values,
                                light_curve_approx['flux'].values - light_curve_approx['flux_err'].values,
                                light_curve_approx['flux'].values + light_curve_approx['flux_err'].values,
                         color=self.plot.colors[passband], alpha=0.2, label=str(passband) + ' approx sigma')
        
            if plot_peak:
                self.plot.plot_sum_passbands(self.t_approx, self.flux_approx, self.flux_err_approx, self.passband_approx, ax)
                self.plot.plot_peak(self.t_approx, self.flux_approx, self.flux_err_approx, self.passband_approx, ax)


 
    def plot_one_graph(self, *, plot_approx = True, passband=None, ax=None, true_peak=None, plot_peak=False, title="", save=None, n_obs = 100):
        """
        """

        if ax is None:
            ax = self.plot._ax_adjust()
            
        if passband is not None:
            self._plot_one_graph_passband(self.t_train, self.flux_train, self.flux_err_train, self.passband_train, passband, ax, plot_approx, plot_peak, n_obs)

        else:
            for band in self.passband2lam.keys():
                #print(band, self.t_train)
                self._plot_one_graph_passband(self.t_train, self.flux_train, self.flux_err_train, self.passband_train, band, ax, plot_approx, plot_peak, n_obs)

        if true_peak is not None:
            self.plot.plot_true_peak(true_peak, ax)

        ax.set_title(title, size=35, pad = 15)
        ax.legend(loc='best', ncol=3, fontsize=20)
        if save is not None:
            plt.savefig(save + ".pdf", format='pdf')
        