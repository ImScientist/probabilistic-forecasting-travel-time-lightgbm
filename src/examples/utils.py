import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy.stats as ss
import jax.scipy.stats as jss
from jax import grad, vmap


def synthetic_data_generation(
        n: int,
        n_dummy_feats: int,
        noise_scale: float,
        seed: int = 40
) -> tuple[pd.DataFrame, list[str], str]:
    """
    Y ~ Gamma(alpha(X), beta(x))
    """

    np.random.seed(seed)

    x = np.random.randn(n, 2 + n_dummy_feats + 1)
    ep = noise_scale * np.random.randn(n, 2)

    a1 = 8. + 3 * np.cos(x[:, 0]) + np.sin(x[:, 1]) + ep[:, 0]
    a2 = .05 * (np.abs(x[:, 0])) + ep[:, 1]

    a1 = softplus(a1)
    a2 = softplus(a2)

    alpha = a1 * a2
    beta = a2

    x[:, -1] = np.random.gamma(shape=alpha, scale=1 / beta, size=n)

    feat_names = [f'col_{i}' for i in range(2 + n_dummy_feats)]
    target = 'y'

    x = pd.DataFrame(
        data=x,
        columns=[f'col_{i}' for i in range(2 + n_dummy_feats)] + [target])
    x['alpha'] = alpha
    x['beta'] = beta

    return x, feat_names, target


def mle_fit(y):
    """ MLE of the parameters of a Gamma(alpha, beta) distribution

    mean = alpha / beta
    std  = sqrt(alpha) / beta

    alpha = (mean/std) ** 2
    beta  = mean / std**2
    """

    # naive estimation using (mean, std)
    y_mean = y.mean()
    y_std = y.std()

    alpha_naive = (y_mean / y_std) ** 2
    beta_naive = y_mean / y_std ** 2
    scale_naive = 1 / beta_naive

    bounds = dict(
        a=(alpha_naive / 5, alpha_naive * 5),
        scale=(scale_naive / 5, scale_naive * 5),
        loc=(0, 0))

    res = ss.fit(dist=ss.gamma, data=y, bounds=bounds)

    assert res.success, "MLE not successful"

    alpha_mle = res.params.a
    beta_mle = 1 / res.params.scale

    return alpha_mle, beta_mle


def softplus(x):
    """ Softplus fn """

    return np.log(1 + np.exp(x))


def softplus_inv(x):
    """ Inverse softplus fn """

    return np.log(-1 + np.exp(x))


def get_rv(a) -> ss.rv_continuous:
    """ Use the raw predictions of the LightGBM model to generate
    Gamma-distributed random variable """

    # (n, 2)
    z = softplus(a)

    alpha = z[:, 0] * z[:, 1]
    beta = z[:, 1]

    return ss.gamma(a=alpha, scale=1 / beta)


def predict_quantiles(booster: lgb.Booster, x: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """ Quantiles of the winning bid distribution """

    assert all([0 <= q <= 1 for q in quantiles])

    # -> (len(quantiles), 1)
    q = np.array(quantiles).reshape(-1, 1)

    rv = get_rv(a=booster.predict(x))

    # -> (len(quantiles), len(x))
    values = rv.ppf(q=q)

    # -> (len(x), len(quantiles))
    return values.T


def gamma_logpdf(x, a1, a2):
    """ Gamma log-pdf

    alpha = a1 * a2
    beta  = a2
    """

    return jss.gamma.logpdf(x, a=a1 * a2, loc=0, scale=1 / a2)


d_gamma_d1 = vmap(grad(gamma_logpdf, argnums=1))
d_gamma_d2 = vmap(grad(gamma_logpdf, argnums=2))

d_gamma_d11 = vmap(grad(grad(gamma_logpdf, argnums=1), argnums=1))
d_gamma_d22 = vmap(grad(grad(gamma_logpdf, argnums=2), argnums=2))


# y_true, y_pred
def custom_loss_lgbm(y, a):
    """ The custom loss is proportional to the negative log-likelihood """

    a = a.reshape((y.size, -1), order='F')
    a = softplus(a)

    return 'log-loss', -float(gamma_logpdf(y, a[:, 0], a[:, 1]).mean()), False


def mae(y, a):
    """ Mean absolute error btw predicted distribution mean and observed value """

    a = a.reshape((y.size, -1), order='F')
    a = softplus(a)

    return 'mae', float(np.abs(a[:, 0] - y).mean()), False


# y_true, y_pred
def custom_objective_lgbm(y, a):
    """ Derive gradient and diagonal of the Hessian matrix """

    # (n, 2)
    a = softplus(a)

    # (n, 2)
    grad_ = np.zeros_like(a)
    hess_ = np.zeros_like(a)

    grad_[:, 0] = -d_gamma_d1(y, a[:, 0], a[:, 1])
    grad_[:, 1] = -d_gamma_d2(y, a[:, 0], a[:, 1])

    hess_[:, 0] = -d_gamma_d11(y, a[:, 0], a[:, 1])
    hess_[:, 1] = -d_gamma_d22(y, a[:, 0], a[:, 1])

    return grad_, hess_
