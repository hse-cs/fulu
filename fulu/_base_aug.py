from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from fulu.plotting import LcPlotter


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
        self.plotter = LcPlotter(passband2lam)
        self.t_train = None
        self.flux_train = None
        self.flux_err_train = None
        self.passband_train = None

    @abstractmethod
    def fit(self, t, flux, flux_err, passband):
        """
        Fit an augmentation model.

        This base method must be called from child classes

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
        self.t_train = np.asarray(t)
        self.flux_train = np.asarray(flux)
        self.flux_err_train = np.asarray(flux_err)
        self.passband_train = np.asarray(passband)

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

        return t_aug, flux_aug, flux_err_aug, passband_aug

    def _plot_passband(self, passband, ax, approx=None):
        """
        Helper to construct a light curve in the next method plotter.

        Parameters:
        -----------
        passband : str or int or float
            A key of self.passband2lam dict.
        ax : matplotlib.pyplot.subplot object
            You can set the axis as an element of your matplotlib.pyplot.figure object.
        approx : tuple of array-like or None
            Augumentated light curve.
        """

        self.plotter.errorbar_passband(
            t=self.t_train,
            flux=self.flux_train,
            flux_err=self.flux_err_train,
            passbands=self.passband_train,
            passband=passband,
            ax=ax,
        )

        if approx:
            anobject_approx = self.plotter._make_dataframe(*approx)
            anobject_approx = anobject_approx.sort_values("time")
            light_curve_approx = anobject_approx[anobject_approx.passband == passband]
            ax.plot(
                light_curve_approx["time"].values,
                light_curve_approx["flux"].values,
                linewidth=3.5,
                color=self.plotter.colors[passband],
                label="{} approx flux".format(passband),
                zorder=10,
            )
            ax.fill_between(
                light_curve_approx["time"].values,
                light_curve_approx["flux"].values - light_curve_approx["flux_err"].values,
                light_curve_approx["flux"].values + light_curve_approx["flux_err"].values,
                color=self.plotter.colors[passband],
                alpha=0.2,
                label="{} approx sigma".format(passband),
            )

    def plot(
        self,
        *,
        plot_approx=True,
        n_approx=1000,
        passband=None,
        ax=None,
        true_peak=None,
        plot_peak=False,
        title="",
        save=None,
    ):
        """
        Plotting train points of light curve with errors for all passbands on one graph by default. A black solid curve isn't plotted at the predicted points. The predicted flux errors are also plotted using a gray bar.

        If you submit the name passband, only the submitted passband will built.

        Parameters:
        -----------
        plot_approx : bool, optional
            Flag indicating it is required to plot an approximation curve or
            isn't
        n_approx : int
            Number of approximation points in each passband.
        passband : optional
            Key of self.passband2lam dict or `None`, plot all passbands in the
            latter case
        ax : matplotlib.pyplot.subplot or None, optional
            `matplotlib` axis to plot the graph on, if `None` then new figure
            will be created and corresponded axis object will be returned
        true_peak : float or None, optional
            Plot the real peak time moment of the light curve
        plot_peak : bool, optional
            Plot approximated model peak
        title : str, optional
            The name of the graph set by `ax`.
        save : str or None, optional
            The path for saving graph.

        Returns
        -------
        ax : matplotlib.pyplot.subplot
            `matplotlib` axis with the graph
        """

        if ax is None:
            ax = self.plotter._ax_adjust()

        approx = self.augmentation(np.min(self.t_train), np.max(self.t_train), n_approx)
        if plot_approx:
            plot_approx = approx

        if passband is None:
            for band in self.passband2lam:
                self._plot_passband(band, ax, plot_approx)
        else:
            self._plot_passband(passband, ax, plot_approx)

        if true_peak is not None:
            self.plotter.plot_true_peak(true_peak=true_peak, ax=ax)

        if plot_peak:
            self.plotter.plot_sum_passbands(*approx, ax=ax)
            self.plotter.plot_peak(*approx, ax=ax)

        ax.set_title(title, size=35, pad=15)
        ax.legend(loc="best", ncol=3, fontsize=20)
        if save is not None:
            self.plotter._save_fig(save)

        return ax
