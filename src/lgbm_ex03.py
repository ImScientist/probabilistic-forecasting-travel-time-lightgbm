"""
    Use lgbm DASK API on real data
"""

import lightgbm as lgb
import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask_ml.model_selection import train_test_split

import utils

DATA_DIR = '/Users/ivanova/Documents/projects/probabilistic-forecasting-travel-time/data'


def main():
    """ ... """

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
            f'{DATA_DIR}/preproc_dask_partitioned/',
            columns=num_feats + cat_feats + [target],
            filters=[('month', '==', 1), ('target', '>', 0)])
        .dropna()
        .repartition(npartitions=12))

    df_tr, df_va = train_test_split(
        df, test_size=.1, random_state=40, blockwise=True, shuffle=False)

    model = lgb.DaskLGBMRegressor(
        objective=utils.custom_objective_lgbm,
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
        eval_metric=[utils.custom_loss_lgbm],
        feature_name=feats,
        categorical_feature=[])

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
