"""
    Use lgbm API
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as ss

import jax.numpy as jnp
import jax.scipy.stats as jss

import lightgbm as lgb
import matplotlib.pyplot as plt
from jax import random, jit, grad, jacobian, jacfwd, jacrev, vmap
from sklearn.model_selection import train_test_split


def data_generation(
        n: int,
        n_dummy_feats: int,
        noise_scale: float,
        seed: int = 40
):
    """
    Y ~ Gamma(alpha(X), beta(x))
    """

    np.random.seed(seed)

    x = np.random.randn(n, 2 + n_dummy_feats)
    ep = noise_scale * np.random.randn(n, 2)

    a1 = 8. + 3 * np.cos(x[:, 0]) + np.sin(x[:, 1]) + ep[:, 0]
    a2 = .05 * (np.abs(x[:, 0])) + ep[:, 1]

    a1 = softplus(a1)
    a2 = softplus(a2)

    alpha = a1 * a2
    beta = a2

    y = np.random.gamma(shape=alpha, scale=1 / beta, size=n)

    return x, y, alpha, beta


def softplus(x):
    """ Softplus fn """

    return np.log(1 + np.exp(x))


def gamma_logpdf(x, a1, a2):
    """ Gamma log-pdf

    alpha = a1 * a2
    beta  = a2
    """

    return jss.gamma.logpdf(x, a=a1 * a2, loc=0, scale=1 / a2)


# TODO: not used anywhere
def gamma_pdf(x, a1, a2):
    """ Gamma pdf

    alpha = a1 * a2
    beta  = a2
    """

    return jss.gamma.pdf(x, a=a1 * a2, loc=0, scale=1 / a2)


d_gamma_d1 = vmap(grad(gamma_logpdf, argnums=1))
d_gamma_d2 = vmap(grad(gamma_logpdf, argnums=2))

d_gamma_d11 = vmap(grad(grad(gamma_logpdf, argnums=1), argnums=1))
d_gamma_d22 = vmap(grad(grad(gamma_logpdf, argnums=2), argnums=2))


def custom_loss_lgbm(a, ds):
    """ The custom loss is proportional to the negative log-likelihood """

    y = ds.get_label()  # (n,)
    a = a.astype('float32')

    a = a.reshape((y.size, -1), order='F')
    a = softplus(a)

    return 'log-loss', -float(gamma_logpdf(y, a[:, 0], a[:, 1]).mean()), False


def custom_objective_lgbm(a, ds):
    """ ... """

    y = ds.get_label()
    a = a.astype('float32')

    # (n, 2)
    a = a.reshape((y.size, -1), order='F')
    a = softplus(a)

    # (n, 2)
    grad_ = np.zeros_like(a)
    hess_ = np.zeros_like(a)

    grad_[:, 0] = -d_gamma_d1(y, a[:, 0], a[:, 1])
    grad_[:, 1] = -d_gamma_d2(y, a[:, 0], a[:, 1])

    hess_[:, 0] = -d_gamma_d11(y, a[:, 0], a[:, 1])
    hess_[:, 1] = -d_gamma_d22(y, a[:, 0], a[:, 1])

    grad_ = grad_.reshape(-1, order='F')
    hess_ = hess_.reshape(-1, order='F')

    return grad_, hess_


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


def plot_dist_means(
        dist_hat_mean_tr, alpha_tr, beta_tr,
        dist_hat_mean_va, alpha_va, beta_va
):
    """ """

    fig = plt.figure(figsize=(12, 5))
    kwargs = dict(alpha=.1, s=5, )
    plt.subplot(1, 2, 1)
    plt.scatter(
        alpha_tr / beta_tr,
        dist_hat_mean_tr,
        label='training dataset', **kwargs)
    plt.xlabel('dist mean')
    plt.ylabel('dist_hat mean')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(
        alpha_va / beta_va,
        dist_hat_mean_va,
        label='test dataset', **kwargs)
    plt.xlabel('dist mean')
    plt.ylabel('dist_hat mean')
    plt.legend()

    plt.show()

    return fig


LGBM_PARAMETERS_DEFAULT = {
    'n_estimators': 100,
    'num_class': 2,
    'learning_rate': .03,
    'num_leaves': 32,
    'max_depth': 6,
    'min_data_in_leaf': 100,
    'early_stopping_rounds': 10,
    'first_metric_only': True,
    # 'verbose': 0
}


def main():
    """ main """

    n = 5_000
    n_tr = 4_000

    x, y, alpha, beta = data_generation(n=n, n_dummy_feats=0, noise_scale=0)

    x_tr = x[:n_tr]
    y_tr = y[:n_tr]
    x_va = x[n_tr:]
    y_va = y[n_tr:]

    # x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.2, random_state=41)

    ds_tr = lgb.Dataset(
        data=x_tr,
        label=y_tr,
        feature_name=['c0', 'c1', 'c2'])

    ds_va = lgb.Dataset(
        data=x_va,
        label=y_va,
        feature_name=['c0', 'c1', 'c2'],
        reference=ds_tr)

    eval_history = {}

    gbm = lgb.train(
        params={**LGBM_PARAMETERS_DEFAULT, **{'objective': custom_objective_lgbm}},
        train_set=ds_tr,
        valid_names=['train', 'val'],
        valid_sets=[ds_tr, ds_va],
        feval=[custom_loss_lgbm],
        keep_training_booster=True,
        callbacks=[
            lgb.record_evaluation(eval_history),
            lgb.log_evaluation(period=1, show_stdv=False)
        ]
    )

    path = opj(save_dir, 'model.txt')
    gbm.save_model(path)

    path = opj(save_dir, 'model_summary.txt')
    model_summary = pd.DataFrame()
    model_summary['feat_name'] = gbm.feature_name()
    model_summary['fi_gain'] = gbm.feature_importance(importance_type='gain').round()
    model_summary['fi_split'] = gbm.feature_importance(importance_type='split')
    model_summary.to_parquet(path)

    rv_tr_hat = get_rv(a=gbm.predict(x_tr))
    rv_va_hat = get_rv(a=gbm.predict(x_va))

    fig = plot_dist_means(
        dist_hat_mean_tr=rv_tr_hat.mean(),
        alpha_tr=alpha[:n_tr],
        beta_tr=beta[:n_tr],
        dist_hat_mean_va=rv_va_hat.mean(),
        alpha_va=alpha[n_tr:],
        beta_va=beta[n_tr:])
    path = opj(save_dir, 'compare_dist_means.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
