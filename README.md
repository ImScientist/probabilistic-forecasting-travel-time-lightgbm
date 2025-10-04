# probabilistic-forecasting-travel-time-lightgbm

Probabilistic prediction of travel time with lightgbm on a large dataset

### Local development

- Launch jupyter lab
  ```shell
  # activate the environment (if you have already created it) 
  source .venv/bin/activate

  jupyter lab
  ```

### Setup Dask cluster locally

- Add helm chart:
  ```shell
  helm repo add dask https://helm.dask.org/
  helm repo update
  ```

- Install Dask on Kubernetes for a single user with Jupyter and dask-kubernetes:
  ```shell
  kubectl create ns dask

  EXTRA_PIP_PACKAGES="lightgbm[dask]==4.6.0 scikit-learn==1.7.2 scipy==1.15.3 jax==0.6.2 jaxlib==0.6.2 matplotlib==3.10.6"
  JUPYTERLAB_ARGS="--config /usr/local/etc/jupyter/jupyter_notebook_config.py"
  
  # Dry run
  helm install -n dask --debug --dry-run my-dask-release dask/dask

  helm install -n dask my-dask-release dask/dask \
    --version 2024.1.1 \
    --set worker.replicas=2 \
    --set-json 'worker.env=[{"name":"EXTRA_PIP_PACKAGES","value":"'${EXTRA_PIP_PACKAGES}'"}]' \
    --set-json 'scheduler.env=[{"name":"EXTRA_PIP_PACKAGES","value":"'${EXTRA_PIP_PACKAGES}'"}]' \
    --set-json 'jupyter.env=[
          {"name":"EXTRA_PIP_PACKAGES","value":"'${EXTRA_PIP_PACKAGES}'"},                   
          {"name":"JUPYTERLAB_ARGS","value":"'${JUPYTERLAB_ARGS}'"}
  ]'
  ```
  It will create the following resources:
    - Jupyter (service + deployment)
    - Scheduler (service + deployment)
    - Two workers (deployment)
    - ...

  You can check if all components are running with: `kubectl -n dask get all`. Since the readiness and liveness probes
  are not implemented you might see that all deployments and services are ready even though they are not. For example,
  the extra python packages specified with `EXTRA_PIP_PACKAGES` might be still being installed in each pod. You can
  check the logs of each pod with `kubectl -n dask logs <pod name>`.


- The Jupyter notebook server and Dask scheduler expose external services to which you can connect to manage
  notebooks, or connect directly to the Dask cluster. You can get these addresses by running the following command:

  ```shell
  export DASK_SCHEDULER="127.0.0.1"
  export DASK_SCHEDULER_UI_IP="127.0.0.1"
  export JUPYTER_NOTEBOOK_IP="127.0.0.1"
  
  export DASK_SCHEDULER_PORT=8080
  export DASK_SCHEDULER_UI_PORT=8081
  export JUPYTER_NOTEBOOK_PORT=8082

  echo tcp://$DASK_SCHEDULER:$DASK_SCHEDULER_PORT               -- Dask Client connection
  echo http://$DASK_SCHEDULER_UI_IP:$DASK_SCHEDULER_UI_PORT     -- Dask dashboard
  echo http://$JUPYTER_NOTEBOOK_IP:$JUPYTER_NOTEBOOK_PORT       -- Jupyter notebook
  
  kubectl port-forward --namespace dask svc/my-dask-release-scheduler $DASK_SCHEDULER_PORT:8786 &
  kubectl port-forward --namespace dask svc/my-dask-release-scheduler $DASK_SCHEDULER_UI_PORT:80 &
  kubectl port-forward --namespace dask svc/my-dask-release-jupyter $JUPYTER_NOTEBOOK_PORT:80
  ```

  The default password to login to the notebook server is `dask`.


- To remove the release execute:
  ```shell
  helm uninstall -n dask my-dask-release
  ```

### Hello world example

- Execute the following snippet in jupyter lab:
  ```python
  import os
  import dask.array as da
  from dask.distributed import Client
  

  address = os.environ['DASK_SCHEDULER_ADDRESS']
  # 'my-dask-release-scheduler:8786'
  # 'tcp://my-dask-release-scheduler.dask.svc.cluster.local:8786'
  
  client = Client(address)
  
  x = da.random.random(size=(2_000, 2_000), chunks=(500, 500))
  y = x + x.T - x.mean(axis=0)
  y = y.persist()
  
  client.restart()
  ```
