#!/usr/bin/env sh

echo """
######################################################################
# Create the following resources in Google cloud:                    #
#  - Bucket in cloud storage                                         #
#  - Service account (and key) with access to GCS                    #
#  - Kubernetes cluster                                              #
######################################################################
"""


echo """
Enable relevant services ...
"""
gcloud services enable \
    artifactregistry.googleapis.com \
    container.googleapis.com \
    cloudbuild.googleapis.com \
    iam.googleapis.com


echo """
Create bucket ...
"""
gcloud storage buckets create gs://$BUCKET_NAME --location=$REGION


echo """
Create a service account with GCS access
Create a service account key
"""
acc_name=dask-svc-account

gcloud iam service-accounts create ${acc_name}

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:${acc_name}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/storage.admin"

gcloud iam service-accounts keys create $DASK_SA_CREDENTIALS \
  --iam-account=${acc_name}@${PROJECT_ID}.iam.gserviceaccount.com


echo """
Create a Kubernetes cluster (it will take some time)
"""
gcloud container clusters create dask-cluster \
  --num-nodes=3 \
  --machine-type=n2-highmem-2 \
  --zone=$ZONE \
  --service-account=${acc_name}@${PROJECT_ID}.iam.gserviceaccount.com
