"""
    Train a probabilistic LightGBM model with custom objective function using Dask

     Store the trained model in a local directory and, eventually, export it to a GCS bucket.
"""

import os
import logging
import numpy as np
import scipy.stats as ss
import jax.scipy.stats as jss

from jax import grad, vmap
import lightgbm as lgb
import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask.distributed import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


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


def plot_eval_history(eval_history, metric: str = 'log-loss'):
    """ Plot training and validation loss curves """

    n = len(eval_history['train'][metric])

    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(n),
             np.array(eval_history['train'][metric]),
             label=f'{metric} train')
    plt.plot(np.arange(n),
             np.array(eval_history['val'][metric]),
             label=f'{metric} val')
    plt.legend()

    return fig


def plot_quantiles(
        qs,
        observed_fractions_tr,
        observed_fractions_va
):
    """ Compare predicted with observed fractions """

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(qs, observed_fractions_tr, s=20, marker='x')
    plt.plot([0, 1], [0, 1], color='grey', alpha=.4, label='training data')
    plt.xlabel('predicted quantiles')
    plt.ylabel('observed fractions')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(qs, observed_fractions_va, s=20, marker='x')
    plt.plot([0, 1], [0, 1], color='grey', alpha=.4, label='validation data')
    plt.xlabel('predicted quantiles')
    plt.ylabel('observed fractions')
    plt.legend()

    return fig


def plot_means_vs_std(
        y_tr_hat_mean,
        y_tr_hat_std,
        y_va_hat_mean,
        y_va_hat_std
):
    """ Compare the predicted means with predicted STDs """

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(y_tr_hat_mean, y_tr_hat_std, s=5, alpha=.3, label='training data')
    plt.xlabel('dist mean')
    plt.ylabel('dist std')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_va_hat_mean, y_va_hat_std, s=5, alpha=.3, label='validation data')
    plt.xlabel('dist mean')
    plt.ylabel('dist std')
    plt.legend()

    return fig


def plot_true_vs_predicted_dist_means(
        dist_hat_mean_tr,
        dist_mean_tr,
        dist_hat_mean_va,
        dist_mean_va
):
    """ Compare predicted with true distribution means used in the
    data generation process (possible only for the synthetically generated data)
    """

    fig = plt.figure(figsize=(12, 5))
    kwargs = dict(alpha=.1, s=5, )

    plt.subplot(1, 2, 1)
    xy_min, xy_max = dist_mean_tr.min(), dist_mean_tr.max()

    plt.scatter(
        dist_mean_tr,
        dist_hat_mean_tr,
        label='training dataset', **kwargs)
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color='grey', alpha=.3)
    plt.xlabel('dist mean')
    plt.ylabel('dist_hat mean')
    plt.legend()

    plt.subplot(1, 2, 2)
    xy_min, xy_max = dist_mean_va.min(), dist_mean_va.max()

    plt.scatter(
        dist_mean_va,
        dist_hat_mean_va,
        label='test dataset', **kwargs)
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color='grey', alpha=.3)
    plt.xlabel('dist mean')
    plt.ylabel('dist_hat mean')
    plt.legend()

    plt.show()

    return fig


if __name__ == '__main__':
    client = Client(address=os.environ['DASK_SCHEDULER_ADDRESS'])

    BUCKET_NAME = 'artifacts-bbd92fb15ef4637aae71c609'

    data_dir = f'gs://{BUCKET_NAME}/data'
    data_dir_preprocessed = os.path.join(data_dir, 'preprocessed')

    num_feats = [
        'trip_distance', 'time',
        'pickup_lon', 'pickup_lat', 'pickup_area',
        'dropoff_lon', 'dropoff_lat', 'dropoff_area']
    cat_feats = [
        'passenger_count', 'vendor_id', 'weekday', 'month']
    feats = num_feats + cat_feats
    target = 'target'
    init_score_feats = ['mean_mle', 'beta_mle']

    data_filters_tr = [('year', '==', 2016), ('month', '==', 1), ('target', '>', 0)]
    data_filters_va = [('year', '==', 2017), ('month', '==', 1), ('target', '>', 0)]

    #
    # Determine init score
    #
    y = (
        dd.read_parquet(
            path=f'{data_dir_preprocessed}',
            columns=[target],
            filters=data_filters_tr)
        .dropna()
        .sample(frac=.015)
        .to_dask_array()
        .compute()
        .reshape(-1))

    y_scaler = 1 / np.median(y)
    y_norm = y_scaler * y

    alpha_mle, beta_mle = mle_fit(y_norm)
    mean_mle = alpha_mle / beta_mle

    logger.info(f'y_scaler = {y_scaler}\n'
                f'alpha_mle = {alpha_mle}\n'
                f'beta_mle = {beta_mle}\n'
                f'mean_mle = {mean_mle}')

    # TODO: remove this snippet after fixing dask-ml
    df_tr = (
        dd.read_parquet(
            path=f'{data_dir_preprocessed}',
            columns=num_feats + cat_feats + [target],
            filters=data_filters_tr)
        .dropna()
        .repartition(npartitions=12)
        .sample(frac=.9)
        .assign(
            mean_mle=mean_mle,
            beta_mle=beta_mle,
            target_norm=lambda x: x[target] * y_scaler)
        .persist())

    df_va = (
        dd.read_parquet(
            path=f'{data_dir_preprocessed}',
            columns=num_feats + cat_feats + [target],
            filters=data_filters_va)
        .dropna()
        .repartition(npartitions=12)
        .sample(frac=.05)
        .assign(
            mean_mle=mean_mle,
            beta_mle=beta_mle,
            target_norm=lambda x: x[target] * y_scaler)
        .persist())

    model = lgb.DaskLGBMRegressor(
        objective=custom_objective_lgbm,
        num_class=2,
        metric=None,
        boosting_type='gbdt',
        tree_learner='data',

        n_estimators=100,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=7,
        min_child_samples=100,

        # 'early_stopping_rounds': 10,  # It seems that early stopping is not supported yet
        # 'first_metric_only': True,
        # reg_alpha=.0,
        # reg_lambda=.0001
    )

    model.fit(
        X=df_tr[feats],
        y=df_tr['target_norm'],
        init_score=df_tr[init_score_feats],

        eval_set=[(df_tr[feats], df_tr['target_norm']),
                  (df_va[feats], df_va['target_norm'])],
        eval_names=['train', 'val'],
        eval_init_score=[df_tr[init_score_feats],
                         df_va[init_score_feats]],

        eval_metric=[custom_loss_lgbm],
        feature_name=feats,
        categorical_feature=cat_feats,
    )

    # Save the model
    local_model = model.to_local()
    booster = local_model.booster_
    booster.save_model('model.txt')

    # Plot training/validation loss
    plot_eval_history(model.evals_result_)
    plt.show()

    # Compare predicted with true distribution mean
    raw_hat_tr = (
            model.predict(X=df_tr[feats]) +
            df_tr[init_score_feats].to_dask_array()
    ).compute()

    raw_hat_va = (
            model.predict(X=df_va[feats]) +
            df_va[init_score_feats].to_dask_array()
    ).compute()

    y_tr = df_tr['target_norm'].compute().values
    y_va = df_va['target_norm'].compute().values

    rv_tr_hat = get_rv(a=raw_hat_tr)
    rv_va_hat = get_rv(a=raw_hat_va)

    qs = np.linspace(.1, .9, 9).reshape(-1, 1)

    # -> (n_qs, n)
    y_tr_hat_qs = rv_tr_hat.ppf(q=qs)
    y_va_hat_qs = rv_va_hat.ppf(q=qs)

    # (n_qs, n) -> (n_qs,)
    observed_fractions_tr = (y_tr_hat_qs > y_tr).mean(axis=1)
    observed_fractions_va = (y_va_hat_qs > y_va).mean(axis=1)

    plot_quantiles(
        qs=qs,
        observed_fractions_tr=observed_fractions_tr,
        observed_fractions_va=observed_fractions_va)
    plt.show()

    client.close()
