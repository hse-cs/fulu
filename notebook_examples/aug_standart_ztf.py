import numpy as np
from fulu import gp_aug
import pandas as pd
from sklearn.model_selection import train_test_split


passband2lam = {0: 1, 1: 2}  # green, red
color = {1: "red", 0: "green"}


def get_object(df, name_in_BTSdf):
    """df - csv with all obj"""
    assert isinstance(name_in_BTSdf, str), "Попробуйте ввести название объекта из ZTF"
    if name_in_BTSdf[:2] == "ZT":
        df_num = df[df.object_id == name_in_BTSdf]
        return df_num
    else:
        return None


def get_passband(anobject, passband):
    light_curve = anobject[anobject.passband == passband]
    return light_curve


def compile_obj(t, flux, flux_err, passband):
    obj = pd.DataFrame()
    obj["mjd"] = t
    obj["flux"] = flux
    obj["flux_err"] = flux_err
    obj["passband"] = passband
    return obj


def augum_gp(anobject, kernel, flag_err=False, n_obs=2000):
    # anobject = get_object(data, name)

    anobject_train, anobject_test = train_test_split(anobject, test_size=0.5, random_state=11)
    model = gp_aug.GaussianProcessesAugmentation(passband2lam)

    model.fit(
        anobject_train["mjd"].values,
        anobject_train["flux"].values,
        anobject_train["flux_err"].values,
        anobject_train["passband"].values,
        kernel,
        flag_err,
    )

    # predict flux for unseen observations
    flux_pred, flux_err_pred = model.predict(anobject_test["mjd"].values, anobject_test.passband, copy=True)

    # augmentation
    t_aug, flux_aug, flux_err_aug, passband_aug = model.augmentation(
        anobject["mjd"].min(), anobject["mjd"].max(), n_obs=n_obs
    )

    anobject_test_pred = compile_obj(
        anobject_test["mjd"].values, flux_pred, flux_err_pred, anobject_test["passband"].values
    )
    anobject_aug = compile_obj(t_aug, flux_aug, flux_err_aug, passband_aug)
    return anobject_test, anobject_test_pred, anobject_aug, flux_pred, anobject_train


def plot_light_curves_ax_band(anobject_test, anobject_train, anobject_approx, ax1, ax2, title=""):
    anobject_test = anobject_test.sort_values("mjd")
    anobject_train = anobject_train.sort_values("mjd")
    anobject_approx = anobject_approx.sort_values("mjd")
    for passband, ax in zip(range(2), [ax1, ax2]):
        light_curve_test = get_passband(anobject_test, passband)
        light_curve_train = get_passband(anobject_train, passband)
        light_curve_approx = get_passband(anobject_approx, passband)
        ax.plot(
            light_curve_approx["mjd"].values,
            light_curve_approx["flux"].values,
            linewidth=5.5,
            color="black",
            label="approx flux",
            zorder=10,
        )
        ax.errorbar(
            light_curve_test["mjd"].values,
            light_curve_test["flux"].values,
            yerr=light_curve_test["flux_err"].values,
            linewidth=3.5,
            marker="o",
            elinewidth=1.7,
            markersize=16.50,
            markeredgecolor="black",
            markeredgewidth=1.50,
            fmt=".",
            color=color[passband],
            label=color[passband][0] + " test",
        )
        ax.errorbar(
            light_curve_train["mjd"].values,
            light_curve_train["flux"].values,
            yerr=light_curve_train["flux_err"].values,
            linewidth=3.5,
            marker="^",
            elinewidth=1.7,
            markersize=14.50,
            markeredgecolor="black",
            markeredgewidth=1.50,
            fmt=".",
            color=color[passband],
            label=color[passband][0] + " train",
        )
        ax.fill_between(
            light_curve_approx["mjd"].values,
            light_curve_approx["flux"].values - light_curve_approx["flux_err"].values,
            light_curve_approx["flux"].values + light_curve_approx["flux_err"].values,
            color="gray",
            alpha=0.2,
            label="approx sigma",
        )
    ax1.set_xlabel("Modified Julian Date", size=30)
    ax1.set_ylabel("Flux [100*uJy]", size=30)
    ax1.grid(linewidth=2)
    ax1.set_title(title, size=35)
    ax2.set_xlabel("Modified Julian Date", size=30)
    ax2.set_ylabel("Flux [100*uJy]", size=30)
    ax2.grid(linewidth=2)
    ax2.set_title(title, size=35)
