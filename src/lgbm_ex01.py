"""
    Use lgbm API
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

import utils

opj = os.path.join


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

    a1 = utils.softplus(a1)
    a2 = utils.softplus(a2)

    alpha = a1 * a2
    beta = a2

    y = np.random.gamma(shape=alpha, scale=1 / beta, size=n)

    return x, y, alpha, beta


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


def main(save_dir: str):
    """ main """

    n = 5_000
    n_tr = 4_000

    x, y, alpha, beta = data_generation(n=n, n_dummy_feats=0, noise_scale=0)

    x_tr = x[:n_tr]
    y_tr = y[:n_tr]
    x_va = x[n_tr:]
    y_va = y[n_tr:]

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
        params={**LGBM_PARAMETERS_DEFAULT,
                **{'objective': utils.custom_objective_lgbm}},
        train_set=ds_tr,
        valid_names=['train', 'val'],
        valid_sets=[ds_tr, ds_va],
        feval=[utils.custom_loss_lgbm],
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

    rv_tr_hat = utils.get_rv(a=gbm.predict(x_tr))
    rv_va_hat = utils.get_rv(a=gbm.predict(x_va))

    fig = plot_dist_means(
        dist_hat_mean_tr=rv_tr_hat.mean(),
        alpha_tr=alpha[:n_tr],
        beta_tr=beta[:n_tr],
        dist_hat_mean_va=rv_va_hat.mean(),
        alpha_va=alpha[n_tr:],
        beta_va=beta[n_tr:])
    path = opj(save_dir, 'compare_dist_means.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
