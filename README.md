# Welcome to Fulu

[![PyPI version](https://badge.fury.io/py/fulu.svg)](https://badge.fury.io/py/fulu)
[![Downloads](https://pepy.tech/badge/fulu)](https://pepy.tech/project/fulu)
[![Tests](https://github.com/HSE-LAMBDA/fulu/actions/workflows/tests.yml/badge.svg)](https://github.com/HSE-LAMBDA/fulu/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`Fulu` is a python library of methods for astronomical light curves approximation based on machine learning. It was named after the variable star Zeta Cassiopeiae 590 light-years from the Sun and officially named [Fulu](https://simbad.cds.unistra.fr/simbad/sim-id?Ident=HR153).


![](https://raw.githubusercontent.com/HSE-LAMBDA/fulu/master/images/cas.png)
_Cassiopeia constellation [[source]](https://en.wikipedia.org/wiki/File:Cassiopeia_constellation_map.svg)_

The library contains our implementation of light curve approximation method based on Gaussian Processes described in [1], and several other methods based on Normalizing Flows, Shallow and Bayesian Neural Networks considered in [2].

- [1] K. Boone.  “Avocado: Photometric Classification of Astronomical Transients with Gaussian Process Augmentation.” The Astronomical Journal (2019). [[journal]](https://doi.org/10.3847/1538-3881/ab5182)[[arxiv]](https://doi.org/10.48550/arXiv.1907.04690)
- [2] M. Demianenko, E. Samorodova, M. Sysak, A. Shiriaev, K. Malanchev, D. Derkach, M. Hushchyn. "Supernova Light Curves Approximation based on Neural Network Models." ArXiv abs/2206.13306 (2022). [[arxiv]](https://doi.org/10.48550/arXiv.2206.13306)



## Install

```sh
pip install fulu
```
or

```sh
python3 -m pip install git+https://github.com/HSE-LAMBDA/fulu
```

## Basic usage

```python
import numpy as np
import fulu

# generate a light curve
passband2lam = {'u': np.log10(3751.36), 'g': np.log10(4741.64), 'r': np.log10(6173.23)}
n_per_band = 10
n = n_per_band * len(passband2lam)
t = np.linspace(0.0, n-1, n)
flux = 10.0 + np.sin(2*t) + np.random.normal(0, 0.1, len(t))
flux_err = 0.1 * np.ones_like(flux)
passbands = np.tile(list(passband2lam), n_per_band)

# approximation
aug = fulu.GaussianProcessesAugmentation(passband2lam)
aug.fit(t, flux, flux_err, passbands)

# augmentation
t_aug, flux_aug, flux_err_aug, passband_aug = aug.augmentation(t.min(), t.max(), 100)

# visualization
plotic = fulu.LcPlotter(passband2lam)
plotic.plot_one_graph_all(t=t, flux=flux, flux_err=flux_err, passbands=passbands,
                          t_approx=t_aug, flux_approx=flux_aug,
                          flux_err_approx=flux_err_aug, passband_approx=passband_aug)
```
![](https://raw.githubusercontent.com/HSE-LAMBDA/fulu/master/images/ex.png)

Please find a plotting example in [`notebooks_examples/plotting.ipynb`](notebooks_examples/plotting.ipynb)
