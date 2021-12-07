import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


class LcPlotter:
    def __init__(self, passband2lam):
        self.passband2lam = passband2lam
        self.colors = {key: tone for key, tone in zip(passband2lam.keys(), mcolors.TABLEAU_COLORS.keys())}

    @staticmethod
    def _make_dataframe(t, flux, flux_err, passband):
        if t is None or flux is None or flux_err is None or passband is None:
            raise ValueError("All values must be numerical arrays")

        obj = pd.DataFrame()
        obj["time"] = t
        obj["flux"] = flux
        obj["flux_err"] = flux_err
        obj["passband"] = passband
        return obj

    @staticmethod
    def _ax_adjust():
        fig = plt.figure(figsize=(20, 10), dpi=100)
        plt.rcParams.update({"font.size": 30})
        fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.15)
        ax = plt.subplot(1, 1, 1)
        ax.spines["bottom"].set_linewidth(3)
        ax.spines["top"].set_linewidth(3)
        ax.spines["left"].set_linewidth(3)
        ax.spines["right"].set_linewidth(3)
        ax.set_xlabel("Time", size=30)
        ax.set_ylabel("Flux", size=30)
        ax.grid(linewidth=2)

        return ax

    @staticmethod
    def _save_fig(save_path):
        plt.savefig(save_path)
        print("Your graph was saved into {}".format(save_path))

    def errorbar_passband(
        self, t, flux, flux_err, passbands, passband, *, ax=None, title="", label="", marker="^", save=None
    ):
        if ax is None:
            ax = self._ax_adjust()

        anobject = self._make_dataframe(t, flux, flux_err, passbands)
        anobject = anobject.sort_values("time")
        light_curve = anobject[anobject["passband"] == passband]

        ax.errorbar(
            light_curve["time"].values,
            light_curve["flux"].values,
            yerr=light_curve["flux_err"].values,
            linewidth=3.5,
            fmt=marker,
            elinewidth=1.7,
            markersize=14.50,
            markeredgecolor="black",
            markeredgewidth=1.50,
            color=self.colors[passband],
            label="{} {}".format(passband, label),
        )

        ax.legend(loc="best", ncol=3, fontsize=20)
        ax.set_title(title, size=35, pad=15)

        if save is not None:
            self._save_fig(save)

        return ax

    def plot_sum_passbands(
        self, t_approx, flux_approx, flux_err_approx, passband_approx, *, ax=None, title="", save=None
    ):
        """ """

        if ax is None:
            ax = self._ax_adjust()
        if (
            (t_approx is not None)
            & (flux_approx is not None)
            & (flux_err_approx is not None)
            & (passband_approx is not None)
        ):
            anobject_approx = self._make_dataframe(t_approx, flux_approx, flux_err_approx, passband_approx)
            anobject_approx = anobject_approx.sort_values("time")
            curve = anobject_approx[["time", "flux"]].groupby("time", as_index=False).sum()
            ax.plot(curve["time"].values, curve["flux"].values, label="sum", linewidth=5.5, color="pink")
            ax.legend(loc="best", ncol=3, fontsize=20)
            ax.set_title(title, size=35, pad=15)
        if save is not None:
            self._save_fig(save)

        return ax

    def plot_peak(self, t_approx, flux_approx, flux_err_approx, passband_approx, *, ax=None, title="", save=None):
        """ """

        if ax is None:
            ax = self._ax_adjust()
        if (
            (t_approx is not None)
            & (flux_approx is not None)
            & (flux_err_approx is not None)
            & (passband_approx is not None)
        ):
            anobject_approx = self._make_dataframe(t_approx, flux_approx, flux_err_approx, passband_approx)
            curve = anobject_approx[["time", "flux"]].groupby("time", as_index=False).sum()
            pred_peak = curve["time"][curve["flux"].argmax()]
            ax.axvline(pred_peak, label="pred peak", color="red", linestyle="--", linewidth=5.5)
            ax.legend(loc="best", ncol=3, fontsize=20)
            ax.set_title(title, size=35, pad=15)
        if save is not None:
            self._save_fig(save)

        return ax

    def plot_approx(
        self, t_approx, flux_approx, flux_err_approx, passband_approx, *, passband, ax=None, title="", save=None
    ):
        """ """

        if ax is None:
            ax = self._ax_adjust()

        anobject_approx = self._make_dataframe(t_approx, flux_approx, flux_err_approx, passband_approx)
        anobject_approx = anobject_approx.sort_values("time")
        light_curve_approx = anobject_approx[anobject_approx.passband == passband]
        ax.plot(
            light_curve_approx["time"].values,
            light_curve_approx["flux"].values,
            linewidth=3.5,
            color=self.colors[passband],
            label="{} approx flux".format(passband),
            zorder=10,
        )
        ax.fill_between(
            light_curve_approx["time"].values,
            light_curve_approx["flux"].values - light_curve_approx["flux_err"].values,
            light_curve_approx["flux"].values + light_curve_approx["flux_err"].values,
            color=self.colors[passband],
            alpha=0.2,
            label="{} approx sigma".format(passband),
        )
        ax.legend(loc="best", ncol=3, fontsize=20)
        ax.set_title(title, size=35, pad=15)
        if save is not None:
            self._save_fig(save)

        return ax

    def plot_one_graph_passband(
        self,
        *,
        t,
        flux,
        flux_err,
        passbands,
        passband,
        ax=None,
        t_test=None,
        flux_test=None,
        flux_err_test=None,
        passband_test=None,
        t_approx=None,
        flux_approx=None,
        flux_err_approx=None,
        passband_approx=None,
        title="",
        save=None,
    ):
        """ """
        if ax is None:
            ax = self._ax_adjust()

        self.errorbar_passband(t=t, flux=flux, flux_err=flux_err, passbands=passbands, passband=passband, ax=ax)
        if (t_test is not None) & (flux_test is not None) & (flux_err_test is not None) & (passband_test is not None):
            self.errorbar_passband(t=t_test, flux=flux_test, flux_err=flux_err_test, passbands=passband_test)

        if (
            (t_approx is not None)
            & (flux_approx is not None)
            & (flux_err_approx is not None)
            & (passband_approx is not None)
        ):
            self.plot_approx(
                t_approx=t_approx,
                flux_approx=flux_approx,
                flux_err_approx=flux_err_approx,
                passband_approx=passband_approx,
                passband=passband,
                ax=ax,
            )

        ax.legend(loc="best", ncol=3, fontsize=20)
        ax.set_title(title, size=35, pad=15)
        if save is not None:
            self._save_fig(save)

        return ax

    def plot_true_peak(self, *, true_peak, ax=None, title="", save=None):
        """ """

        if ax is None:
            ax = self._ax_adjust()

        ax.axvline(true_peak, label="true peak", color="black", linewidth=5.5)
        ax.set_title(title, size=35, pad=15)
        if save is not None:
            self._save_fig(save)

        return ax

    def plot_one_graph_all(
        self,
        *,
        t,
        flux,
        flux_err,
        passbands,
        t_approx=None,
        flux_approx=None,
        flux_err_approx=None,
        passband_approx=None,
        passband=None,
        ax=None,
        true_peak=None,
        plot_peak=False,
        title="",
        save=None,
    ):
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
            ax = self._ax_adjust()

        if passband is not None:
            self.plot_one_graph_passband(
                t=t,
                flux=flux,
                flux_err=flux_err,
                passbands=passbands,
                passband=passband,
                ax=ax,
                t_approx=t_approx,
                flux_approx=flux_approx,
                flux_err_approx=flux_err_approx,
                passband_approx=passband_approx,
            )

        else:
            for band in self.passband2lam.keys():
                self.plot_one_graph_passband(
                    t=t,
                    flux=flux,
                    flux_err=flux_err,
                    passbands=passbands,
                    passband=band,
                    ax=ax,
                    t_approx=t_approx,
                    flux_approx=flux_approx,
                    flux_err_approx=flux_err_approx,
                    passband_approx=passband_approx,
                )

        if true_peak is not None:
            self.plot_true_peak(true_peak, ax=ax)

        if plot_peak:
            self.plot_sum_passbands(
                t_approx=t_approx,
                flux_approx=flux_approx,
                flux_err_approx=flux_err_approx,
                passband_approx=passband_approx,
                ax=ax,
            )
            self.plot_peak(
                t_approx=t_approx,
                flux_approx=flux_approx,
                flux_err_approx=flux_err_approx,
                passband_approx=passband_approx,
                ax=ax,
            )

        ax.set_title(title, size=35, pad=15)
        ax.legend(loc="best", ncol=3, fontsize=20)
        if save is not None:
            self._save_fig(save)

        return ax
