"""
    Use lgbm DASK API
"""

import os
import logging
import numpy as np
import lightgbm as lgb
import dask.dataframe as dd
import matplotlib.pyplot as plt
from distributed import Client, LocalCluster

import utils
import utils_plot

logger = logging.getLogger(__name__)
opj = os.path.join


def main_synthetic(save_dir: str):
    """ main """

    cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit='6GB')
    client = Client(cluster)

    n = 5_000
    n_tr = 4_000

    xy, feat_names, target = utils.synthetic_data_generation(
        n=n, n_dummy_feats=0, noise_scale=0)

    alpha_mle, beta_mle = utils.mle_fit(y=xy[target].values[:n_tr])
    mean_mle = alpha_mle / beta_mle

    # Add `mean_mle`, `beta_mle` as columns to xy
    init_score_names = ['mean_mle', 'beta_mle']
    xy['mean_mle'] = mean_mle
    xy['beta_mle'] = beta_mle

    xy_tr = dd.from_pandas(xy.iloc[:n_tr], npartitions=4)
    xy_va = dd.from_pandas(xy.iloc[n_tr:], npartitions=4)

    # metric='None',  # Disable default metrics
    model = lgb.DaskLGBMRegressor(
        objective=utils.custom_objective_lgbm,
        num_class=2,
        metric=None,
        boosting_type='gbdt',
        tree_learner='data',

        n_estimators=60,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=100,

        # 'early_stopping_rounds': 10,  # It seems that early stopping is not supported yet
        # 'first_metric_only': True,
        # reg_alpha=.0,
        # reg_lambda=.0001
    )

    model.fit(
        X=xy_tr[feat_names],
        y=xy_tr[target],
        init_score=xy_tr[init_score_names],

        eval_set=[(xy_tr[feat_names], xy_tr[target]),
                  (xy_va[feat_names], xy_va[target])],
        eval_names=['train', 'val'],
        eval_init_score=[xy_tr[init_score_names],
                         xy_va[init_score_names]],

        eval_metric=[utils.custom_loss_lgbm],
        feature_name=feat_names,
        categorical_feature=[],
    )

    # Save the model
    local_model = model.to_local()
    booster = local_model.booster_
    booster.save_model('model.txt')

    # Plot training/validation loss
    path = opj(save_dir, 'eval_history.png')
    fig = utils_plot.plot_eval_history(model.evals_result_)
    fig.savefig(path, dpi=200, bbox_inches='tight')

    # Compare predicted with true distribution mean
    raw_hat_tr = (
            model.predict(xy_tr[feat_names]) +
            xy_tr[init_score_names].to_dask_array()
    ).compute()

    raw_hat_va = (
            model.predict(xy_va[feat_names]) +
            xy_va[init_score_names].to_dask_array()
    ).compute()

    rv_tr_hat = utils.get_rv(a=raw_hat_tr)
    rv_va_hat = utils.get_rv(a=raw_hat_va)

    utils_plot.plot_true_vs_predicted_dist_means(
        dist_hat_mean_tr=rv_hat_tr.mean(),  # noqa
        dist_mean_tr=(xy_tr['alpha'] / xy_tr['beta']).compute(),
        dist_hat_mean_va=rv_hat_va.mean(),  # noqa
        dist_mean_va=(xy_va['alpha'] / xy_va['beta']).compute())
    plt.show()

    qs = np.linspace(.1, .9, 9).reshape(-1, 1)

    # -> (n_qs, n)
    y_tr_hat_qs = rv_tr_hat.ppf(q=qs)
    y_va_hat_qs = rv_va_hat.ppf(q=qs)

    y_tr = xy_tr[target].compute().values
    y_va = xy_va[target].compute().values

    # (n_qs, n) -> (n_qs,)
    observed_fractions_tr = (y_tr_hat_qs > y_tr).mean(axis=1)
    observed_fractions_va = (y_va_hat_qs > y_va).mean(axis=1)

    _ = utils_plot.plot_quantiles(
        qs=qs,
        observed_fractions_tr=observed_fractions_tr,
        observed_fractions_va=observed_fractions_va)
    plt.show()

    # Compare predicted means with predicted STDs
    y_tr_hat_mean = np.array(rv_tr_hat.mean())
    y_va_hat_mean = np.array(rv_va_hat.mean())

    y_tr_hat_std = rv_tr_hat.std()
    y_va_hat_std = rv_va_hat.std()

    rng = np.random.default_rng(12)
    idx_tr = rng.choice(np.arange(len(y_tr_hat_mean)), size=(1_000,))
    idx_va = rng.choice(np.arange(len(y_va_hat_mean)), size=(1_000,))

    _ = utils_plot.plot_means_vs_std(
        y_tr_hat_mean=y_tr_hat_mean[idx_tr],
        y_tr_hat_std=y_tr_hat_std[idx_tr],
        y_va_hat_mean=y_va_hat_mean[idx_va],
        y_va_hat_std=y_va_hat_std[idx_va])
    plt.show()

    client.close()


def main_nyc(
        data_dir_preprocessed: str,
        save_dir: str
):
    """ Use NYC dataset """

    cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit='6GB')
    client = Client(cluster)

    num_feats = [
        'trip_distance', 'time',
        'pickup_lon', 'pickup_lat', 'pickup_area',
        'dropoff_lon', 'dropoff_lat', 'dropoff_area']
    cat_feats = [
        'passenger_count', 'vendor_id', 'weekday', 'month']
    feats = num_feats + cat_feats
    target = 'target'
    init_score_feats = ['mean_mle', 'beta_mle']

    data_filters = [('month', '==', 1), ('target', '>', 0)]

    y = (
        dd.read_parquet(
            path=f'{data_dir_preprocessed}/train',
            columns=[target],
            filters=data_filters)
        .dropna()
        .sample(frac=.015)
        .to_dask_array()
        .compute())

    y_scaler = 1 / np.median(y)
    y_norm = y_scaler * y

    alpha_mle, beta_mle = utils.mle_fit(y_norm)
    mean_mle = alpha_mle / beta_mle

    logger.info(f'y_scaler = {y_scaler}\n'
                f'alpha_mle = {alpha_mle}\n'
                f'beta_mle = {beta_mle}\n'
                f'mean_mle = {mean_mle}')

    # TODO: save the results
    df_tr = (
        dd.read_parquet(
            path=f'{data_dir_preprocessed}/train',
            columns=num_feats + cat_feats + [target],
            filters=data_filters)
        .dropna()
        .repartition(npartitions=12)
        .sample(frac=.1)
        .assign(
            mean_mle=mean_mle,
            beta_mle=beta_mle,
            target_norm=lambda x: x[target] * y_scaler)
        .persist())

    df_va = (
        dd.read_parquet(
            path=f'{data_dir_preprocessed}/validation',
            columns=num_feats + cat_feats + [target],
            filters=data_filters)
        .dropna()
        .repartition(npartitions=12)
        .sample(frac=.1)
        .assign(
            mean_mle=mean_mle,
            beta_mle=beta_mle,
            target_norm=lambda x: x[target] * y_scaler)
        .persist())

    model = lgb.DaskLGBMRegressor(
        objective=utils.custom_objective_lgbm,
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

        eval_metric=[utils.custom_loss_lgbm],
        feature_name=feats,
        categorical_feature=cat_feats,
    )

    # Save the model
    local_model = model.to_local()
    booster = local_model.booster_
    booster.save_model('model.txt')

    # Plot training/validation loss
    path = opj(save_dir, 'eval_history.png')
    fig = utils_plot.plot_eval_history(model.evals_result_)
    fig.savefig(path, dpi=200, bbox_inches='tight')

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

    rv_tr_hat = utils.get_rv(a=raw_hat_tr)
    rv_va_hat = utils.get_rv(a=raw_hat_va)

    qs = np.linspace(.1, .9, 9).reshape(-1, 1)

    # -> (n_qs, n)
    y_tr_hat_qs = rv_tr_hat.ppf(q=qs)
    y_va_hat_qs = rv_va_hat.ppf(q=qs)

    # (n_qs, n) -> (n_qs,)
    observed_fractions_tr = (y_tr_hat_qs > y_tr).mean(axis=1)
    observed_fractions_va = (y_va_hat_qs > y_va).mean(axis=1)

    #
    _ = utils_plot.plot_quantiles(
        qs=qs,
        observed_fractions_tr=observed_fractions_tr,
        observed_fractions_va=observed_fractions_va)
    plt.show()

    client.close()
