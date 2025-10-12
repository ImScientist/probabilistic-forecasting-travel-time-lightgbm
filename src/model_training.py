import os

import logging
import numpy as np
import jax.scipy.stats as jss
from jax import grad, vmap
import lightgbm as lgb
import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask_ml.model_selection import train_test_split
from dask.distributed import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


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


def softplus(x):
    """ Softplus fn """

    return np.log(1 + np.exp(x))


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


# y_true, y_pred
def custom_loss_lgbm(y, a):
    """ The custom loss is proportional to the negative log-likelihood """

    # y = ds.get_label()  # (n,)
    # a = a.astype('float32')

    a = a.reshape((y.size, -1), order='F')
    a = softplus(a)

    return 'log-loss', -float(gamma_logpdf(y, a[:, 0], a[:, 1]).mean()), False


if __name__ == '__main__':
    DATA_DIR = 'gs://data-55acf5c126ac2d4fd4c09d61/data'

    data_dir_preprocessed = os.path.join(DATA_DIR, 'preprocessed')

    client = Client(address=os.environ['DASK_SCHEDULER_ADDRESS'])

    num_feats = [
        'trip_distance', 'time',
        'pickup_lon', 'pickup_lat', 'pickup_area',
        'dropoff_lon', 'dropoff_lat', 'dropoff_area']

    cat_feats = [
        'passenger_count', 'vendor_id', 'weekday', 'month']

    feats = num_feats + cat_feats

    target = 'target'

    df = (
        dd.read_parquet(
            data_dir_preprocessed,
            columns=num_feats + cat_feats + [target],
            filters=[('month', '==', 1), ('target', '>', 0)])
        .dropna()
        .repartition(npartitions=12))

    df_tr, df_va = train_test_split(
        df, test_size=.1, random_state=40, blockwise=True, shuffle=False)

    model = lgb.DaskLGBMRegressor(
        objective=custom_objective_lgbm,
        num_class=2,
        metric='None',
        boosting_type='gbdt',
        tree_learner='data',

        n_estimators=60,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=100,
        reg_alpha=.0,
        reg_lambda=.0001)

    model.fit(
        df_tr[feats],
        df_tr[target],

        eval_set=[(df_tr[feats], df_tr[target]),
                  (df_va[feats], df_va[target])],
        eval_names=['train', 'val'],
        eval_metric=[custom_loss_lgbm],
        feature_name=feats,
        categorical_feature=cat_feats)

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
