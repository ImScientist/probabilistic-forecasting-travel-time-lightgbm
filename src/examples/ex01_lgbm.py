"""
    Use lgbm API
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb

import utils
import utils_plot

opj = os.path.join

LGBM_PARAMETERS_DEFAULT = {
    'n_estimators': 100,
    'num_class': 2,
    'learning_rate': .025,
    'num_leaves': 64,
    'max_depth': 7,
    'min_data_in_leaf': 100,
    'early_stopping_rounds': 10,
    'first_metric_only': True,
    # 'verbose': 0
}


def create_model_summary(gbm: lgb.Booster) -> pd.DataFrame:
    """ Create model summary """

    model_summary = pd.DataFrame()
    model_summary['feat_name'] = gbm.feature_name()
    model_summary['fi_gain'] = gbm.feature_importance(importance_type='gain').round()
    model_summary['fi_split'] = gbm.feature_importance(importance_type='split')

    return model_summary


def custom_loss(a, ds):
    """ The custom loss is proportional to the negative log-likelihood """

    return utils.custom_loss_lgbm(y=ds.get_label(), a=a)


def custom_objective(a, ds):
    """ The custom objective is proportional to the negative log-likelihood """

    # (n,)
    y = ds.get_label()

    # -> (n, 2)
    a = a.astype('float32').reshape((y.size, -1), order='F')

    grad, hess = utils.custom_objective_lgbm(y, a)

    grad = grad.reshape(-1, order='F')
    hess = hess.reshape(-1, order='F')

    return grad, hess


def main_synthetic(save_dir: str):
    """ Training with synthetic data """

    n = 5_000
    n_tr = 4_000

    xy, feat_names, target = utils.synthetic_data_generation(
        n=n, n_dummy_feats=0, noise_scale=0)

    xy_tr = xy.iloc[:n_tr]
    xy_va = xy.iloc[n_tr:]

    alpha_mle, beta_mle = utils.mle_fit(y=xy_tr[target].values)
    mean_mle = alpha_mle / beta_mle

    init_score_tr = np.ones(shape=(len(xy_tr), 2)) * np.array([[mean_mle, beta_mle]])
    init_score_va = np.ones(shape=(len(xy_va), 2)) * np.array([[mean_mle, beta_mle]])

    ds_tr = lgb.Dataset(
        data=xy_tr[feat_names],
        label=xy_tr[target],
        init_score=init_score_tr,
        feature_name=feat_names)

    ds_va = lgb.Dataset(
        data=xy_va[feat_names],
        label=xy_va[target],
        init_score=init_score_va,
        feature_name=feat_names,
        reference=ds_tr)

    eval_history = {}

    gbm = lgb.train(
        params={**LGBM_PARAMETERS_DEFAULT,
                **{'objective': custom_objective}},
        train_set=ds_tr,
        valid_names=['train', 'val'],
        valid_sets=[ds_tr, ds_va],
        feval=[custom_loss],
        keep_training_booster=True,
        callbacks=[
            lgb.record_evaluation(eval_history),
            lgb.log_evaluation(period=1, show_stdv=False)
        ]
    )

    path = opj(save_dir, 'model.txt')
    gbm.save_model(path)

    path = opj(save_dir, 'eval_history.png')
    fig = utils_plot.plot_eval_history(eval_history)
    fig.savefig(path, dpi=200, bbox_inches='tight')

    path = opj(save_dir, 'model_summary.parquet')
    model_summary = create_model_summary(gbm=gbm)
    model_summary.to_parquet(path)

    raw_hat_tr = gbm.predict(xy_tr[feat_names]) + init_score_tr
    raw_hat_va = gbm.predict(xy_va[feat_names]) + init_score_va

    rv_tr_hat = utils.get_rv(a=raw_hat_tr)
    rv_va_hat = utils.get_rv(a=raw_hat_va)

    fig = utils_plot.plot_true_vs_predicted_dist_means(
        dist_hat_mean_tr=rv_tr_hat.mean(),
        dist_mean_tr=xy_tr['alpha'] / xy_tr['beta'],
        dist_hat_mean_va=rv_va_hat.mean(),
        dist_mean_va=xy_va['alpha'] / xy_va['beta'])
    path = opj(save_dir, 'compare_dist_means.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')

    qs = np.linspace(.1, .9, 9).reshape(-1, 1)

    # -> (n_qs, n)
    y_tr_hat_qs = rv_tr_hat.ppf(q=qs)
    y_va_hat_qs = rv_va_hat.ppf(q=qs)

    y_tr = xy_tr[target].values
    y_va = xy_va[target].values

    # (n_qs, n) -> (n_qs,)
    observed_fractions_tr = (y_tr_hat_qs > y_tr).mean(axis=1)
    observed_fractions_va = (y_va_hat_qs > y_va).mean(axis=1)

    fig = utils_plot.plot_quantiles(
        qs=qs,
        observed_fractions_tr=observed_fractions_tr,
        observed_fractions_va=observed_fractions_va)
    path = opj(save_dir, 'compare_quantiles.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')


def main_nyc(save_dir: str):
    """ Training with NYC data """

    DATA_DIR = '/Users/ivanova/Documents/projects/probabilistic-forecasting-travel-time/data'

    LGBM_PARAMETERS_DEFAULT = {
        'n_estimators': 200,
        'num_class': 2,
        'learning_rate': .05,
        'num_leaves': 64,
        'max_depth': 7,
        'min_data_in_leaf': 100,
        'early_stopping_rounds': 10,
        'first_metric_only': True,
        # 'verbose': 0
    }

    num_feats = [
        'trip_distance', 'time',
        'pickup_lon', 'pickup_lat', 'pickup_area',
        'dropoff_lon', 'dropoff_lat', 'dropoff_area']

    cat_feats = [
        'passenger_count', 'vendor_id', 'weekday', 'month']

    feats = num_feats + cat_feats

    df_tr = pd.read_parquet(f'{DATA_DIR}/preprocessed/train/data_2016-01.parquet')
    df_va = pd.read_parquet(f'{DATA_DIR}/preprocessed/validation/data_2016-01.parquet')

    target_scaler = 1 / df_tr['target'].median()
    df_tr['target_norm'] = df_tr['target'] * target_scaler
    df_va['target_norm'] = df_va['target'] * target_scaler

    alpha_mle, beta_mle = utils.mle_fit(y=df_tr['target_norm'].sample(n=100_000).values)
    mean_mle = alpha_mle / beta_mle

    init_score_tr = np.ones(shape=(len(df_tr), 2)) * np.array([[mean_mle, beta_mle]])
    init_score_va = np.ones(shape=(len(df_va), 2)) * np.array([[mean_mle, beta_mle]])

    ds_tr = lgb.Dataset(
        data=df_tr.loc[:, feats],
        label=df_tr['target_norm'].values,
        init_score=init_score_tr,
        feature_name=feats,
        categorical_feature=cat_feats)

    ds_va = lgb.Dataset(
        data=df_va.loc[:, feats],
        label=df_va['target_norm'].values,
        init_score=init_score_va,
        feature_name=feats,
        categorical_feature=cat_feats,
        reference=ds_tr)

    eval_history = {}

    gbm = lgb.train(
        params={**LGBM_PARAMETERS_DEFAULT,
                **{'objective': custom_objective}},
        train_set=ds_tr,
        valid_names=['train', 'val'],
        valid_sets=[ds_tr, ds_va],
        feval=[custom_loss],
        keep_training_booster=True,
        callbacks=[
            lgb.record_evaluation(eval_history),
            lgb.log_evaluation(period=1, show_stdv=False)
        ]
    )

    # Evaluation
    path = opj(save_dir, 'model.txt')
    gbm.save_model(path)

    path = opj(save_dir, 'eval_history.png')
    fig = utils_plot.plot_eval_history(eval_history)
    fig.savefig(path, dpi=200, bbox_inches='tight')

    path = opj(save_dir, 'model_summary.parquet')
    model_summary = create_model_summary(gbm=gbm)
    model_summary.to_parquet(path)

    rv_tr_hat = utils.get_rv(a=gbm.predict(df_tr[feats]) + init_score_tr)
    rv_va_hat = utils.get_rv(a=gbm.predict(df_va[feats]) + init_score_va)

    qs = np.linspace(.1, .9, 9).reshape(-1, 1)

    # -> (n_qs, n)
    y_tr_hat_qs = rv_tr_hat.ppf(q=qs)
    y_va_hat_qs = rv_va_hat.ppf(q=qs)

    # (n_qs, n) -> (n_qs,)
    observed_fractions_tr = (y_tr_hat_qs > df_tr['target_norm'].values).mean(axis=1)
    observed_fractions_va = (y_va_hat_qs > df_va['target_norm'].values).mean(axis=1)

    fig = utils_plot.plot_quantiles(
        qs=qs,
        observed_fractions_tr=observed_fractions_tr,
        observed_fractions_va=observed_fractions_va)
    path = opj(save_dir, 'compare_quantiles.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')

    # Compare predicted means with predicted STDs
    y_tr_hat_mean = np.array(rv_tr_hat.mean())
    y_va_hat_mean = np.array(rv_va_hat.mean())

    y_tr_hat_std = rv_tr_hat.std()
    y_va_hat_std = rv_va_hat.std()

    rng = np.random.default_rng(12)
    idx_tr = rng.choice(np.arange(len(y_tr_hat_mean)), size=(1_000,))
    idx_va = rng.choice(np.arange(len(y_va_hat_mean)), size=(1_000,))

    fig = utils_plot.plot_means_vs_std(
        y_tr_hat_mean=y_tr_hat_mean[idx_tr],
        y_tr_hat_std=y_tr_hat_std[idx_tr],
        y_va_hat_mean=y_va_hat_mean[idx_va],
        y_va_hat_std=y_va_hat_std[idx_va])
    path = opj(save_dir, 'compare_means_with_stds.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
