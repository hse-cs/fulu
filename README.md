# Fulu
## Light curve approximation

[![Tests](https://github.com/HSE-LAMBDA/fulu/actions/workflows/tests.yml/badge.svg)](https://github.com/HSE-LAMBDA/fulu/actions/workflows/tests.yml)

### Install
```sh
python3 -m pip install git+https://github.com/HSE-LAMBDA/fulu
```

### Example
```python
import numpy as np
from fulu import MLPRegressionAugmentation

passband2lam = {'u': np.log10(3751.36), 'g': np.log10(4741.64), 'r': np.log10(6173.23),
                'i': np.log10(7501.62), 'z': np.log10(8679.19), 'y': np.log10(9711.53)}
n_per_band = 10
n = n_per_band * len(passband2lam)
t = np.linspace(0.0, n-1, n)
flux = 10.0 + np.sin(t)
flux_err = np.ones_like(flux)
passbands = np.tile(list(passband2lam), n_per_band)

aug = MLPRegressionAugmentation(passband2lam)
aug.fit(t, flux, flux_err, passbands)
t_aug, flux_aug, flux_err_aug, passband_aug = aug.augmentation(t.min(), t.max(), 100)
```
