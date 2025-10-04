"""
    Use lgbm DASK API
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd

import lightgbm as lgb
import matplotlib.pyplot as plt

import utils


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

    a1 = utils.softplus(a1)
    a2 = utils.softplus(a2)

    alpha = a1 * a2
    beta = a2

    x[:, -1] = np.random.gamma(shape=alpha, scale=1 / beta, size=n)

    feat_names = [f'col_{i}' for i in range(2 + n_dummy_feats)]
    target = 'y'

    x = pd.DataFrame(x, columns=[f'col_{i}' for i in range(2 + n_dummy_feats)] + [target])
    x['alpha'] = alpha
    x['beta'] = beta

    return x, feat_names, target


def plot_dist_means(
        dist_hat_mean_tr: np.ndarray,
        alpha_tr: np.ndarray,
        beta_tr: np.ndarray,
        dist_hat_mean_va: np.ndarray,
        alpha_va: np.ndarray,
        beta_va: np.ndarray
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
        eval_metric=[utils.custom_loss_lgbm],
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

    rv_hat_tr = utils.get_rv(a=raw_hat_tr)
    rv_hat_va = utils.get_rv(a=raw_hat_va)

    plot_dist_means(
        dist_hat_mean_tr=rv_hat_tr.mean(),  # noqa
        alpha_tr=xy_tr['alpha'].compute(),
        beta_tr=xy_tr['beta'].compute(),
        dist_hat_mean_va=rv_hat_va.mean(),  # noqa
        alpha_va=xy_va['alpha'].compute(),
        beta_va=xy_va['beta'].compute())
    plt.show()
