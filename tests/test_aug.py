import numpy as np
import pytest

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


@pytest.mark.parametrize("cls", subclasses(BaseAugmentation))
def test_aug_with_sample_data(cls):
    passband2lam, *lc = sample_data()
    t, *_ = lc
    aug = cls(passband2lam)
    aug.fit(*lc)
    aug.augmentation(t[0], t[-1], 100)
