import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

from fulu import GaussianProcessesAugmentation
from fulu._base_aug import BaseAugmentation


def subclasses(cls):
    return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in subclasses(c))


def sample_data():
    passband2lam = {'u': np.log10(3751.36), 'g': np.log10(4741.64), 'r': np.log10(6173.23),
                    'i': np.log10(7501.62), 'z': np.log10(8679.19), 'y': np.log10(9711.53)}
    n_per_band = 10
    n = n_per_band * len(passband2lam)
    t = np.linspace(0.0, n-1, n)
    flux = 10.0 + np.sin(t)
    flux_err = np.ones_like(flux)
    passbands = np.tile(list(passband2lam), n_per_band)
    return passband2lam, t, flux, flux_err, passbands


@pytest.mark.parametrize(
    ("cls", "init_kwargs"),
    [(cls, {}) for cls in subclasses(BaseAugmentation)]
    + [
        (GaussianProcessesAugmentation, dict(use_err=True)),
        (GaussianProcessesAugmentation, dict(kernel=ConstantKernel(1.0)*RBF([1, 1]) + Matern() + WhiteKernel())),
    ],
)
def test_aug_with_sample_data(cls, init_kwargs):
    n_aug = 100
    passband2lam, *lc = sample_data()
    n_band = len(passband2lam)
    t, *_ = lc
    aug = cls(passband2lam)
    aug_fit = aug.fit(*lc)
    assert aug_fit is aug, '.fit() must return self'
    t_aug, flux_aug, flux_err_aug, passband_aug = aug.augmentation(t.min(), t.max(), n_aug)
    assert_allclose(t_aug, np.tile(np.linspace(t.min(), t.max(), n_aug), n_band))
    assert flux_aug.size == n_aug * n_band
    assert np.all(flux_aug >= 0.0)
    assert flux_err_aug.size == n_aug * n_band
    assert np.all(flux_err_aug >= 0.0)
    assert passband_aug.size == n_aug * n_band
    assert set(passband_aug) == set(passband2lam)
