"""
    Use lgbm DASK API
"""

import os
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
import scipy.stats as ss

import jax.numpy as jnp
import jax.scipy.stats as jss

import lightgbm as lgb
import matplotlib.pyplot as plt

from jax import random, jit, grad, vmap


# from distributed import Client, LocalCluster, Worker, WorkerPlugin, PipInstall


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

    x = pd.DataFrame(x, columns=[f'col_{i}' for i in range(2 + n_dummy_feats)] + [target])
    x['alpha'] = alpha
    x['beta'] = beta

    return x, feat_names, target


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


def plot_dist_means(
        dist_hat_mean_tr, alpha_tr, beta_tr,
        dist_hat_mean_va, alpha_va, beta_va
):
    """ """

    fig = plt.figure(figsize=(12, 5))
    kwargs = dict(alpha=.1, s=5, )

    plt.subplot(1, 2, 1)
    dist_mean_tr = alpha_tr / beta_tr
    xy_min = min(dist_mean_tr.min(), dist_hat_mean_tr.min())
    xy_max = max(dist_mean_tr.max(), dist_hat_mean_tr.max())

    plt.scatter(
        dist_mean_tr,
        dist_hat_mean_tr,
        label='training dataset', **kwargs)
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color='grey', alpha=.4)
    plt.xlabel('dist mean')
    plt.ylabel('dist_hat mean')
    plt.xlim((xy_min, xy_max))
    plt.ylim((xy_min, xy_max))

    plt.legend()

    plt.subplot(1, 2, 2)
    dist_mean_va = alpha_va / beta_va
    xy_min = min(dist_mean_va.min(), dist_hat_mean_va.min())
    xy_max = max(dist_mean_va.max(), dist_hat_mean_va.max())

    plt.scatter(
        dist_mean_va,
        dist_hat_mean_va,
        label='test dataset', **kwargs)
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color='grey', alpha=.4)
    plt.xlabel('dist mean')
    plt.ylabel('dist_hat mean')
    plt.xlim((xy_min, xy_max))
    plt.ylim((xy_min, xy_max))
    plt.legend()

    plt.show()

    return fig


def main():
    """ main """

    n = 5_000
    n_tr = 4_000

    xy, feat_names, target = data_generation(n=n, n_dummy_feats=0, noise_scale=0)

    xy_tr = dd.from_pandas(xy.iloc[:n_tr], npartitions=4)
    xy_va = dd.from_pandas(xy.iloc[n_tr:], npartitions=4)

    # ds_tr = lgb.Dataset(
    #     data=x_tr,
    #     label=y_tr,
    #     feature_name=['c0', 'c1', 'c2'])
    #
    # ds_va = lgb.Dataset(
    #     data=x_va,
    #     label=y_va,
    #     feature_name=['c0', 'c1', 'c2'],
    #     reference=ds_tr)

    # metric='None',  # Disable default metrics
    model = lgb.DaskLGBMRegressor(
        objective=custom_objective_lgbm,
        num_class=2,
        metric=None,
        boosting_type='gbdt',
        tree_learner='data',

        n_estimators=60,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=100,

        # 'early_stopping_rounds': 10,
        # 'first_metric_only': True,
        # reg_alpha=.0,
        # reg_lambda=.0001
    )

    model.fit(
        xy_tr[feat_names],
        xy_tr[target],

        eval_set=[(xy_tr[feat_names], xy_tr[target]),
                  (xy_va[feat_names], xy_va[target])],
        eval_names=['train', 'val'],
        eval_metric=[custom_loss_lgbm],
        feature_name=feat_names,
        categorical_feature=[],

        # callbacks=[
        #     lgb.log_evaluation(period=1, show_stdv=False)
        # ]
    )

    # Save the model
    local_model = model.to_local()
    booster = local_model.booster_
    booster.save_model('model.txt')

    # Plot training loss
    metric = 'log-loss'
    plt.plot(model.evals_result_['train'][metric], label=f'train {metric}')
    plt.plot(model.evals_result_['val'][metric], label=f'validation {metric}')
    plt.legend()
    plt.show()

    # Compare predicted with true distribution mean
    raw_hat_tr = model.predict(X=xy_tr[feat_names]).compute()
    raw_hat_va = model.predict(X=xy_va[feat_names]).compute()

    rv_hat_tr = get_rv(a=raw_hat_tr)
    rv_hat_va = get_rv(a=raw_hat_va)

    plot_dist_means(
        dist_hat_mean_tr=rv_hat_tr.mean(),
        alpha_tr=xy_tr['alpha'].compute(),
        beta_tr=xy_tr['beta'].compute(),
        dist_hat_mean_va=rv_hat_va.mean(),
        alpha_va=xy_va['alpha'].compute(),
        beta_va=xy_va['beta'].compute())
    plt.show()

    # rv_tr_hat = get_rv(a=model.predict(x_tr))

    # eval_history = {}
    #
    # gbm = lgb.train(
    #     params={**LGBM_PARAMETERS_DEFAULT, **{'objective': custom_objective_lgbm}},
    #     train_set=ds_tr,
    #     valid_names=['train', 'val'],
    #     valid_sets=[ds_tr, ds_va],
    #     feval=[custom_loss_lgbm],
    #     keep_training_booster=True,
    #     callbacks=[
    #         lgb.record_evaluation(eval_history),
    #         lgb.log_evaluation(period=1, show_stdv=False)
    #     ]
    # )

    # path = opj(save_dir, 'model.txt')
    # gbm.save_model(path)
    #
    # path = opj(save_dir, 'model_summary.txt')
    # model_summary = pd.DataFrame()
    # model_summary['feat_name'] = gbm.feature_name()
    # model_summary['fi_gain'] = gbm.feature_importance(importance_type='gain').round()
    # model_summary['fi_split'] = gbm.feature_importance(importance_type='split')
    # model_summary.to_parquet(path)
    #
    # rv_tr_hat = get_rv(a=gbm.predict(x_tr))
    # rv_va_hat = get_rv(a=gbm.predict(x_va))
    #
    # fig = plot_dist_means(
    #     dist_hat_mean_tr=rv_tr_hat.mean(),
    #     alpha_tr=alpha[:n_tr],
    #     beta_tr=beta[:n_tr],
    #     dist_hat_mean_va=rv_va_hat.mean(),
    #     alpha_va=alpha[n_tr:],
    #     beta_va=beta[n_tr:])
    # path = opj(save_dir, 'compare_dist_means.png')
    # fig.savefig(path, dpi=200, bbox_inches='tight')
