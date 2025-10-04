import os
import dask.array as da
import dask.dataframe as dd
import numpy as np
import lightgbm as lgb

from distributed import Client, LocalCluster, Worker, WorkerPlugin, PipInstall
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import mean_absolute_error, mean_squared_error


def client_setup():
    """ """

    plugin = PipInstall(
        packages=["lightgbm[dask]==4.6.0", "scikit-learn==1.2.0",
                  "dask-ml==2023.3.24", "jax==0.4.7", "jaxlib==0.4.7"])

    cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit='6GB')
    client = Client(cluster)
    client.register_plugin(plugin)
