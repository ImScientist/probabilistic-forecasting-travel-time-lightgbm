import numpy as np
import scipy.stats as ss
import jax.scipy.stats as jss
import lightgbm as lgb
from jax import grad, vmap


def softplus(x):
    """ Softplus fn """

    return np.log(1 + np.exp(x))


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

    # y = ds.get_label()  # (n,)
    # a = a.astype('float32')

    a = a.reshape((y.size, -1), order='F')
    a = softplus(a)

    return 'log-loss', -float(gamma_logpdf(y, a[:, 0], a[:, 1]).mean()), False


# y_true, y_pred
def custom_objective_lgbm(y, a):
    """ ... """

    # y = ds.get_label()
    # a = a.astype('float32')

    # (n, 2)
    # a = a.reshape((y.size, -1), order='F')
    # a = a.reshape((y.size, -1))
    a = softplus(a)

    # (n, 2)
    grad_ = np.zeros_like(a)
    hess_ = np.zeros_like(a)

    grad_[:, 0] = -d_gamma_d1(y, a[:, 0], a[:, 1])
    grad_[:, 1] = -d_gamma_d2(y, a[:, 0], a[:, 1])

    hess_[:, 0] = -d_gamma_d11(y, a[:, 0], a[:, 1])
    hess_[:, 1] = -d_gamma_d22(y, a[:, 0], a[:, 1])

    # grad_ = grad_.reshape(-1, order='F')
    # hess_ = hess_.reshape(-1, order='F')

    # grad_ = grad_.reshape(-1)
    # hess_ = hess_.reshape(-1)

    return grad_, hess_
