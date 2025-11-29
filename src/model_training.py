"""
    Train a probabilistic LightGBM model with custom objective function using Dask

    Store the trained model in a local directory and, eventually, export it to a GCS bucket.
"""

import os
import time
import logging
import tarfile
import tempfile
import datetime as dt

import numpy as np
import lightgbm as lgb
import scipy.stats as ss
import scipy.optimize as so
import scipy.special as sspec
import matplotlib.pyplot as plt

import jax.scipy.stats as jss
from jax import grad, vmap

import dask.dataframe as dd
from dask.distributed import Client
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
opj = os.path.join


def mle_loss(params, y, counts):
    """ ... """

    alpha, scale = params

    log_pdf = ss.gamma.logpdf(y, a=alpha, scale=scale)

    return - (counts * log_pdf).sum() / counts.sum()


def mle_fit_new(y, counts):
    """ MLE of the parameters of a Gamma(alpha, beta) distribution

    mean = alpha / beta
    std  = sqrt(alpha) / beta

    alpha = (mean/std) ** 2
    beta  = mean / std**2
    """

    # naive estimation using (mean, std)
    y_mean = (y * counts).sum() / counts.sum()
    y_std = np.sqrt((counts * (y - y_mean) ** 2).sum() / counts.sum())

    alpha_0 = (y_mean / y_std) ** 2
    beta_0 = y_mean / y_std ** 2
    scale_0 = 1 / beta_0

    x0 = np.array([alpha_0, scale_0])

    lb = np.array([1e-3, 1e-3])
    ub = np.array([np.inf, np.inf])
    bounds = so.Bounds(lb=lb, ub=ub)

    res = so.minimize(
        fun=lambda x: mle_loss(x, y, counts),
        x0=x0,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 500},
        tol=1e-8)

    assert res.success, "MLE not successful"

    alpha_mle, scale_mle = res.x
    beta_mle = 1 / scale_mle

    return alpha_mle, beta_mle


def softplus(x):
    """ Softplus fn """

    return np.log(1 + np.exp(x))


def softplus_inv(x):
    """ Inverse softplus fn """

    return np.log(-1 + np.exp(x))


def get_rv(raw_score) -> ss.rv_continuous:
    """ Use the raw predictions of the LightGBM model to generate
    Gamma-distributed random variable """

    # (n, 2)
    z = softplus(raw_score)

    alpha = z[:, 0] * z[:, 1]
    beta = z[:, 1]

    return ss.gamma(a=alpha, scale=1 / beta)


def predict_quantiles(booster: lgb.Booster, x: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """ Quantiles of the winning bid distribution """

    assert all([0 <= q <= 1 for q in quantiles])

    # -> (len(quantiles), 1)
    q = np.array(quantiles).reshape(-1, 1)

    rv = get_rv(raw_score=booster.predict(x))

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


def mae_lightgbm(y, a):
    """ Mean absolute error btw predicted distribution mean and observed value """

    a = a.reshape((y.size, -1), order='F')
    a = softplus(a)

    return 'mae', float(np.abs(a[:, 0] - y).mean()), False


def crps_gamma(y, alpha, beta):
    """ CRPS for a Gamma(alpha, beta) distribution

    CRPS(y) = y * (2 * CDF_gamma(y| alpha, beta) - 1)
              + E[X] * (1 - 2 * CDF_gamma(y| alpha+1, beta))
              - 2 * E[X] * (I_{0.5}(alpha, alpha+1) - .5)
    """

    mean = alpha / beta

    line_1 = y * (2 * ss.gamma.cdf(x=y, a=alpha, scale=1 / beta) - 1)
    line_2 = mean * (1 - 2 * ss.gamma.cdf(x=y, a=alpha + 1, scale=1 / beta))
    line_3 = -2 * mean * (sspec.betainc(alpha, alpha + 1, 0.5) - .5)

    result = line_1 + line_2 + line_3

    return result


def crps_lightgbm(y, a):
    """ Average CRPS
    Inputs adapted to the raw output of the LightGBM model
    """

    a = a.reshape((y.size, -1), order='F')
    a = softplus(a)

    alpha = a[:, 0] * a[:, 1]
    beta = a[:, 1]

    return 'crps', crps_gamma(y, alpha, beta).mean(), False


def rel_std_lightgbm(y, a):
    """ Average relative STD = STD[X] / E[X]
    Inputs adapted to the raw output of the LightGBM model
    """

    raw_score = a.reshape((y.size, -1), order='F')

    rv_tr_hat = get_rv(raw_score=raw_score)

    return 'rel_std', (rv_tr_hat.std() / rv_tr_hat.mean()).mean(), False


def export_dir_to_gcs(
        project_id: str,
        bucket_name: str,
        blob_name: str,
        export_dir: str
):
    """ Compress folder and export it to gcs-blob """

    blob = (storage.Client(project=project_id)
            .bucket(bucket_name=bucket_name)
            .blob(blob_name=blob_name))

    with tempfile.NamedTemporaryFile('w', suffix=".tar.gz") as temp:
        with tarfile.open(temp.name, "w:gz") as temp_tar:
            temp_tar.add(export_dir)

        blob.upload_from_filename(temp.name)


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


def plot_rel_stds(
        y_tr_hat_mean,
        y_tr_hat_std,
        y_va_hat_mean,
        y_va_hat_std
):
    """ Plot a distribution of predicted std / predicted mean """

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(y_tr_hat_std / y_tr_hat_mean, bins=20, alpha=.3, label='training data')
    plt.xlabel('std / mean')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(y_va_hat_std / y_va_hat_mean, bins=20, alpha=.3, label='validation data')
    plt.xlabel('std / mean')
    plt.legend()

    return fig


if __name__ == '__main__':
    client = Client(address=os.environ['DASK_SCHEDULER_ADDRESS'])

    PROJECT_ID = ''
    BUCKET_NAME = 'artifacts-.....'

    save_dir = 'output'
    os.makedirs(save_dir, exist_ok=True)

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
    init_score_feats = ['a1', 'a2']

    data_filters_tr = [('year', '==', 2016), ('month', '==', 1), ('target', '>', 0)]
    data_filters_va = [('year', '==', 2017), ('month', '==', 1), ('target', '>', 0)]

    #
    # Determine init score
    #
    y_values = (
        dd.read_parquet(
            path=f'{data_dir_preprocessed}',
            columns=[target],
            filters=data_filters_tr)
        .dropna())

    y_median = y_values[target].median_approximate().compute()
    y_scaler = 1 / y_median

    # target | n | target_norm
    df_counts = y_values.groupby(target).size().compute().rename('n').reset_index()
    df_counts['target_norm'] = df_counts[target] * y_scaler

    alpha_mle, beta_mle = mle_fit_new(
        y=df_counts['target_norm'].values,
        counts=df_counts['n'].values)

    mean_mle = alpha_mle / beta_mle

    logger.info(f'y_scaler = {y_scaler}\n'
                f'alpha_mle = {alpha_mle}\n'
                f'beta_mle = {beta_mle}\n'
                f'mean_mle = {mean_mle}')

    df_tr = (
        dd.read_parquet(
            path=f'{data_dir_preprocessed}',
            columns=num_feats + cat_feats + [target],
            filters=data_filters_tr)
        .dropna()
        .repartition(npartitions=12)
        # .sample(frac=.9)
        .assign(
            a1=softplus_inv(mean_mle),
            a2=softplus_inv(beta_mle),
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
            a1=softplus_inv(mean_mle),
            a2=softplus_inv(beta_mle),
            target_norm=lambda x: x[target] * y_scaler)
        .persist())

    model = lgb.DaskLGBMRegressor(
        objective=custom_objective_lgbm,
        num_class=2,
        metric=None,
        boosting_type='gbdt',
        tree_learner='data',

        n_estimators=600,
        learning_rate=0.05,
        max_delta_step=1.,
        num_leaves=511,
        max_depth=11,
        min_child_samples=1_000,

        # 'early_stopping_rounds': 10,  # It seems that early stopping is not supported yet
        # 'first_metric_only': True,
        # reg_alpha=.0,
        # reg_lambda=.0001
    )

    ti = time.time()
    model.fit(
        X=df_tr[feats],
        y=df_tr['target_norm'],
        init_score=df_tr[init_score_feats],

        eval_set=[(df_tr[feats], df_tr['target_norm']),
                  (df_va[feats], df_va['target_norm'])],
        eval_names=['train', 'val'],
        eval_init_score=[df_tr[init_score_feats],
                         df_va[init_score_feats]],

        eval_metric=[custom_loss_lgbm,
                     mae_lightgbm,
                     crps_lightgbm,
                     rel_std_lightgbm],
        feature_name=feats,
        categorical_feature=cat_feats,
    )
    tf = time.time()
    training_time = tf - ti
    logger.info(f'training time = {training_time}')

    # Save the model
    path = opj(save_dir, 'model.txt')
    local_model = model.to_local()
    booster = local_model.booster_
    booster.save_model(path)

    # Plot training/validation loss
    path = opj(save_dir, 'eval_history_log-loss.png')
    fig = plot_eval_history(model.evals_result_, metric='log-loss')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()

    path = opj(save_dir, 'eval_history_mae.png')
    fig = plot_eval_history(model.evals_result_, metric='mae')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()

    path = opj(save_dir, 'eval_history_crps.png')
    fig = plot_eval_history(model.evals_result_, metric='crps')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()

    path = opj(save_dir, 'eval_history_mean_rel_std.png')
    fig = plot_eval_history(model.evals_result_, metric='rel_std')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()

    # TODO: update num_iteration (pick the value where the val-loss is the lowest)
    # 0 <-> get all trees
    num_iteration = 0

    # Compare predicted with true distribution mean
    df_tr_sample = df_tr.sample(frac=.05).persist()
    df_va_sample = df_va.sample(frac=.90).persist()

    raw_hat_tr = (
            model.predict(X=df_tr_sample[feats], num_iteration=num_iteration) +
            df_tr_sample[init_score_feats].to_dask_array()
    ).compute()

    raw_hat_va = (
            model.predict(X=df_va_sample[feats], num_iteration=num_iteration) +
            df_va_sample[init_score_feats].to_dask_array()
    ).compute()

    y_tr = df_tr_sample['target_norm'].compute().values
    y_va = df_va_sample['target_norm'].compute().values

    rv_tr_hat = get_rv(raw_score=raw_hat_tr)
    rv_va_hat = get_rv(raw_score=raw_hat_va)

    qs = np.linspace(.1, .9, 9).reshape(-1, 1)

    # -> (n_qs, n)
    y_tr_hat_qs = rv_tr_hat.ppf(q=qs)
    y_va_hat_qs = rv_va_hat.ppf(q=qs)

    # (n_qs, n) -> (n_qs,)
    observed_fractions_tr = (y_tr_hat_qs > y_tr).mean(axis=1)
    observed_fractions_va = (y_va_hat_qs > y_va).mean(axis=1)

    path = opj(save_dir, 'calibration plot.png')
    fig = plot_quantiles(
        qs=qs,
        observed_fractions_tr=observed_fractions_tr,
        observed_fractions_va=observed_fractions_va)
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.show()

    #
    # Compare predicted means with predicted STDs
    #

    y_tr_hat_mean = rv_tr_hat.mean()
    y_va_hat_mean = rv_va_hat.mean()

    y_tr_hat_std = rv_tr_hat.std()
    y_va_hat_std = rv_va_hat.std()

    rng = np.random.default_rng(12)
    idx_tr = rng.choice(np.arange(len(y_tr_hat_mean)), size=(1_000,))  # noqa
    idx_va = rng.choice(np.arange(len(y_va_hat_mean)), size=(1_000,))  # noqa

    path = opj(save_dir, 'relative_std.png')
    fig = plot_rel_stds(
        y_tr_hat_mean=y_tr_hat_mean[idx_tr],
        y_tr_hat_std=y_tr_hat_std[idx_tr],
        y_va_hat_mean=y_va_hat_mean[idx_va],
        y_va_hat_std=y_va_hat_std[idx_va])
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.show()

    path = opj(save_dir, 'dist_mean_vs_std.png')
    fig = plot_means_vs_std(
        y_tr_hat_mean=y_tr_hat_mean[idx_tr],
        y_tr_hat_std=y_tr_hat_std[idx_tr],
        y_va_hat_mean=y_va_hat_mean[idx_va],
        y_va_hat_std=y_va_hat_std[idx_va])
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.show()

    export_dir_to_gcs(
        project_id=PROJECT_ID,
        bucket_name=BUCKET_NAME,
        blob_name=f'artifacts_{dt.datetime.now().isoformat()}',
        export_dir=save_dir)

    client.close()
