from abc import ABC, abstractmethod
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


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

def create_colors_dict(passband2lam):
    """
    """
    
    colors = {}
    tableau_colors = mcolors.TABLEAU_COLORS.keys()
    for key, tone in zip(passband2lam.keys(), tableau_colors):
        colors.update([[key, tone]])
    return colors

def compile_obj(t, flux, flux_err, passband):
    """
    """
    
    obj = pd.DataFrame()
    obj['mjd']      = t
    obj['flux']     = flux
    obj['flux_err'] = flux_err
    obj['passband'] = passband
    return obj

def get_passband(anobject, passband):
    """
    """
    
    light_curve = anobject[anobject.passband == passband]
    return light_curve

def get_group_by_mjd(anobject_approx):
    """
    """

    curve = anobject_approx[['mjd', 'flux']].groupby('mjd', as_index=False).sum()
    return curve

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
        self.colors = create_colors_dict(passband2lam)

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
        flux_aug, flux_err_aug = self.predict(t_aug, passband_aug)

        return t_aug, flux_aug, flux_err_aug, passband_aug
    
    def ax_adjust():
        """
        """

        fig = plt.figure(figsize=(20,10), dpi = 50)
        plt.rcParams.update({'font.size': 30})
        fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.15)   
        ax = plt.subplot(1, 1, 1)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['top'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.set_xlabel('Time', size=30)
        ax.set_ylabel('Flux', size=30)
        ax.grid(linewidth=2)

        return ax

    def errorbar_passband(anobject_train, passband, ax, anobject_test=None):
        """
        """

        anobject_train = compile_obj(*anobject_train)
        anobject_train = anobject_train.sort_values('mjd')
        light_curve_train = get_passband(anobject_train, passband)

        ax.errorbar(light_curve_train['mjd'].values, light_curve_train['flux'].values, \
                         yerr=light_curve_train['flux_err'].values, linewidth=3.5, \
                         marker='^', elinewidth=1.7 ,markersize=14.50, \
               markeredgecolor='black', markeredgewidth=1.50, \
                         fmt='.', color=colors[passband], label=str(passband)+' train')
        if anobject_test is not None:
            anobject_test = compile_obj(*anobject_test)
            anobject_test = anobject_test.sort_values('mjd')
            light_curve_test = get_passband(anobject_test, passband)
            ax.errorbar(light_curve_test['mjd'].values, light_curve_test['flux'].values, \
                         yerr=light_curve_test['flux_err'].values, linewidth=3.5, \
                         marker='o', elinewidth=1.7 ,markersize=14.50, \
               markeredgecolor='black', markeredgewidth=1.50, \
                         fmt='.', color=colors[passband], label=str(passband)+' test')

    def plot_approx(anobject_approx, passband, ax):
        """
        """
        
        anobject_approx = compile_obj(*anobject_approx)
        anobject_approx = anobject_approx.sort_values('mjd')
        light_curve_approx = get_passband(anobject_approx, passband)
        ax.plot(light_curve_approx['mjd'].values, light_curve_approx['flux'].values, \
                        linewidth=3.5, color=colors[passband], label=str(passband) + ' approx flux', zorder=10)
        ax.fill_between(light_curve_approx['mjd'].values,\
                                light_curve_approx['flux'].values - light_curve_approx['flux_err'].values, \
                                light_curve_approx['flux'].values + light_curve_approx['flux_err'].values,
                         color=colors[passband], alpha=0.2, label=str(passband) + ' approx sigma')

    def plot_sum_passbands(anobject_approx, ax):
        """
        """

        anobject_approx = compile_obj(*anobject_approx)
        curve = get_group_by_mjd(anobject_approx)
        ax.plot(curve['mjd'].values, curve['flux'].values, label='sum', linewidth=5.5, color='pink')

    def plot_peak(anobject_approx, ax):
        """
        """

        anobject_approx = compile_obj(*anobject_approx)
        curve = get_group_by_mjd(anobject_approx)
        pred_peak_mjd = curve['mjd'][curve['flux'].argmax()]
        ax.axvline(pred_peak_mjd, label='pred peak', color='red', linestyle = '--', linewidth=5.5)

    def plot_one_graph_passband(anobject_train, passband, ax, anobject_test=None, anobject_approx=None, plot_peak=None):
        """
        """

        errorbar_passband(anobject_train, passband, ax, anobject_test)
        if anobject_approx is not None:
            plot_approx(anobject_approx, passband, ax)
            if plot_peak is not None:
                plot_sum_passbands(anobject_approx, ax)
                plot_peak(anobject_approx, ax)

    def plot_true_peak(true_peak_mjd, ax):
        """
        """

        ax.axvline(true_peak_mjd, label='true peak', color='black', linewidth=5.5)

    def plot_one_graph(anobject_train, anobject_test=None, passband=None, anobject_approx=None, ax=None, true_peak_mjd=None, plot_peak=None, title="", save=None):
        """
        Plotting test and train points of light curve with errors for all passbands on one graph by default.

        If you submit the name passband, only the submitted passband is built.

        If you submit augmented data, a black solid curve is plotted at the predicted points.
        The predicted flux errors are also plotted using a gray bar.

        It is assumed that your light curve is a table containing:
        observation time in MJD with the name of the corresponding column "mjd";
        flux in the column called "flux";
        flux errors in the column called "flux_err";
        passbands in the column called "passband".

        Parameters:
        -----------
        anobject_test, anobject_train : list or array-like
            Lists of lists (time, flux, error of flux, passbands) for one object from your dataset after split.
        passband : format in which the name of the passband is specified in your table
        anobject_approx : list or array-like
            List of lists (time, flux, error of flux, passbands) for current object after augmentation.
        ax : matplotlib.pyplot.subplot object
            You can set the axis as an element of your matplotlib.pyplot.figure object.
        true_peak_mjd : float or int
            The real peak of the light curve flux.
        plot_peak : bool or int
            Flag is responsible for plotting peak by max flux of overall flux. 
        title : str
            The name of the graph set by ax.
        save : str
            The name for saving graph (in pdf format).
        """


        if ax is None:
            ax = ax_adjust()

        if passband is not None:
            plot_one_graph_passband(anobject_train, passband, ax, anobject_test, anobject_approx, plot_peak)

        else:
            for band in passband2lam.keys():
                plot_one_graph_passband(anobject_train, band, ax, anobject_test, anobject_approx, plot_peak)

        if true_peak_mjd is not None:
            plot_true_peak(true_peak_mjd, ax)

        ax.set_title(title, size=35)
        ax.legend(loc='best', ncol=3, fontsize=20)
        if save is not None:
            plt.savefig(save + ".pdf", format='pdf')
