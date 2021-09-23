import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


class Plotting_lc():
    
    def __init__(self, passband2lam):
        self.passband2lam = passband2lam
        self.colors = {key: tone for key, tone in zip(passband2lam.keys(), mcolors.TABLEAU_COLORS.keys())}
        
    def _make_dataframe(self, t, flux, flux_err, passband):
        """
        """
        
        if (t is not None)&(flux is not None)&(flux_err is not None)&(passband is not None):
            obj = pd.DataFrame()
            obj['time']      = t
            obj['flux']     = flux
            obj['flux_err'] = flux_err
            obj['passband'] = passband
            return obj
        return None
    
    def _ax_adjust(self, *, title=""):
        """
        """

        fig = plt.figure(figsize=(20,10), dpi = 100)
        plt.rcParams.update({'font.size': 30})
        fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.15)   
        ax = plt.subplot(1, 1, 1)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['top'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.set_xlabel('Time', size=30)
        ax.set_ylabel('Flux', size=30)
        ax.grid(linewidth=2)

        return ax

    def errorbar_passband(self, *, t_train, flux_train, flux_err_train, passband_train, passband, ax=None, t_test=None, flux_test=None, flux_err_test=None, passband_test=None, title=""):
        """
        """
        
        if ax is None:
            ax = self._ax_adjust(title=title)
            
        anobject_train = self._make_dataframe(t_train, flux_train, flux_err_train, passband_train)
        anobject_train = anobject_train.sort_values('time')
        light_curve_train = anobject_train[anobject_train['passband'] == passband]

        ax.errorbar(light_curve_train['time'].values, light_curve_train['flux'].values,
                         yerr=light_curve_train['flux_err'].values, linewidth=3.5,
                         marker='^', elinewidth=1.7 ,markersize=14.50,
               markeredgecolor='black', markeredgewidth=1.50,
                         fmt='.', color=self.colors[passband], label=str(passband)+' train')
        
        anobject_test = self._make_dataframe(t_test, flux_test, flux_err_test, passband_test)
        if anobject_test is not None:
            anobject_test = anobject_test.sort_values('time')
            light_curve_test = anobject_test[anobject_test['passband'] == passband]
            ax.errorbar(light_curve_test['time'].values, light_curve_test['flux'].values,
                         yerr=light_curve_test['flux_err'].values, linewidth=3.5,
                         marker='o', elinewidth=1.7 ,markersize=14.50,
               markeredgecolor='black', markeredgewidth=1.50,
                         fmt='.', color=self.colors[passband], label=str(passband)+' test')
        ax.legend(loc='best', ncol=3, fontsize=20)    
        ax.set_title(title, size=35, pad = 15)
        return ax


    def plot_sum_passbands(self, *, t_approx, flux_approx, flux_err_approx, passband_approx, ax=None, title=""):
        """
        """
        
        if ax is None:
            ax = self._ax_adjust(title=title)
        if (t_approx is not None)&(flux_approx is not None)&(flux_err_approx is not None)&(passband_approx is not None):   
            anobject_approx = self._make_dataframe(t_approx, flux_approx, flux_err_approx, passband_approx)
            anobject_approx = anobject_approx.sort_values('time')
            curve = anobject_approx[['time', 'flux']].groupby('time', as_index=False).sum()
            ax.plot(curve['time'].values, curve['flux'].values, label='sum', linewidth=5.5, color='pink')
            ax.legend(loc='best', ncol=3, fontsize=20)
            ax.set_title(title, size=35, pad = 15)
        return ax

    def plot_peak(self, *, t_approx, flux_approx, flux_err_approx, passband_approx, ax=None, title=""):
        """
        """
        
        if ax is None:
            ax = self._ax_adjust(title=title)
        if (t_approx is not None)&(flux_approx is not None)&(flux_err_approx is not None)&(passband_approx is not None):
            anobject_approx = self._make_dataframe(t_approx, flux_approx, flux_err_approx, passband_approx)
            curve = anobject_approx[['time', 'flux']].groupby('time', as_index=False).sum()
            pred_peak = curve['time'][curve['flux'].argmax()]
            ax.axvline(pred_peak, label='pred peak', color='red', linestyle = '--', linewidth=5.5)
            ax.legend(loc='best', ncol=3, fontsize=20)
            ax.set_title(title, size=35, pad = 15)
        return ax

    def plot_approx(self, *, t_approx, flux_approx, flux_err_approx, passband_approx, passband, ax=None, title=""):
        """
        """
        
        if ax is None:
            ax = self._ax_adjust(title=title)
            
        anobject_approx = self._make_dataframe(t_approx, flux_approx, flux_err_approx, passband_approx)
        anobject_approx = anobject_approx.sort_values('time')
        light_curve_approx = anobject_approx[anobject_approx.passband == passband]
        ax.plot(light_curve_approx['time'].values, light_curve_approx['flux'].values,
                        linewidth=3.5, color=self.colors[passband], label=str(passband) + ' approx flux', zorder=10)
        ax.fill_between(light_curve_approx['time'].values,
                                light_curve_approx['flux'].values - light_curve_approx['flux_err'].values,
                                light_curve_approx['flux'].values + light_curve_approx['flux_err'].values,
                         color=self.colors[passband], alpha=0.2, label=str(passband) + ' approx sigma')
        ax.legend(loc='best', ncol=3, fontsize=20)
        ax.set_title(title, size=35, pad = 15)
        return ax
        
                    
    def plot_one_graph_passband(self, *, t_train, flux_train, flux_err_train, passband_train, passband, ax=None, t_test=None, flux_test=None, flux_err_test=None, passband_test=None, t_approx=None, flux_approx=None, flux_err_approx=None, passband_approx=None, title=""):
        """
        """
        if ax is None:
            ax = self._ax_adjust()
            
        self.errorbar_passband(t_train=t_train, flux_train=flux_train, flux_err_train=flux_err_train, passband_train=passband_train, passband=passband, ax=ax, t_test=t_test, flux_test=flux_test, flux_err_test=flux_err_test, passband_test=passband_test)
        
        if (t_approx is not None)&(flux_approx is not None)&(flux_err_approx is not None)&(passband_approx is not None):
            self.plot_approx(t_approx=t_approx, flux_approx=flux_approx, flux_err_approx=flux_err_approx, passband_approx=passband_approx, passband=passband, ax=ax)
            
        ax.legend(loc='best', ncol=3, fontsize=20)
        ax.set_title(title, size=35, pad = 15)
        return ax

    def plot_true_peak(self, *, true_peak, ax=None, title=""):
        """
        """
        
        if ax is None:
            ax = self._ax_adjust(title=title)
            
        ax.axvline(true_peak, label='true peak', color='black', linewidth=5.5)
        ax.set_title(title, size=35, pad = 15)
        return ax

    
    def plot_one_graph_all(self, *, t_train, flux_train, flux_err_train, passband_train, t_approx=None, flux_approx=None, flux_err_approx=None, passband_approx=None, passband=None, ax=None, true_peak=None, plot_peak=False, title="", save=None):
        """
        Plotting train points of light curve with errors for all passbands on one graph by default.

        If you submit the name passband, only the submitted passband is built.

        If you submit augmented data, a black solid curve is plotted at the predicted points.
        The predicted flux errors are also plotted using a gray bar.

        Parameters:
        -----------
        t_train, t_approx : array-like
            Timestamps of light curve observations, which are used in fit method\after augmentation.
        flux_train, flux_approx : array-like
            Flux of the light curve observations, which are used in fit method\after augmentation.
        flux_err_train, flux_err_approx : array-like
            Flux errors of the light curve observations, which are used in fit method\after augmentation.
        passband_train, passband_approx : array-like
            Passband IDs for each observation, which are used in fit method\after augmentation.
        passband : str or int or float
            Passband ID.
        ax : matplotlib.pyplot.subplot object
            You can set the axis as an element of your matplotlib.pyplot.figure object.
        true_peak : float or int
            The real peak of the light curve flux.
        plot_peak : bool or int
            Flag is responsible for plotting peak by max flux of overall flux. 
        title : str
            The name of the graph set by ax.
        save : str
            The name for saving graph (in pdf format).
        """
        
        if ax is None:
            ax = self._ax_adjust(title=title)
            
        if passband is not None:
            self.plot_one_graph_passband(t_train=t_train, flux_train=flux_train, flux_err_train=flux_err_train, passband_train=passband_train, passband=passband, ax=ax, t_approx=t_approx, flux_approx=flux_approx, flux_err_approx=flux_err_approx, passband_approx=passband_approx)

        else:
            for band in self.passband2lam.keys():
                self.plot_one_graph_passband(t_train=t_train, flux_train=flux_train, flux_err_train=flux_err_train, passband_train=passband_train, passband=band, ax=ax, t_approx=t_approx, flux_approx=flux_approx, flux_err_approx=flux_err_approx, passband_approx=passband_approx)

        if true_peak is not None:
            self.plot_true_peak(true_peak, ax)
        
        if plot_peak:
            self.plot_sum_passbands(t_approx=t_approx, flux_approx=flux_approx, flux_err_approx=flux_err_approx, passband_approx=passband_approx, ax=ax)
            self.plot_peak(t_approx=t_approx, flux_approx=flux_approx, flux_err_approx=flux_err_approx, passband_approx=passband_approx, ax=ax)

        ax.set_title(title, size=35, pad = 15)
        ax.legend(loc='best', ncol=3, fontsize=20)
        if save is not None:
            plt.savefig(save + ".pdf", format='pdf')
        
        return ax