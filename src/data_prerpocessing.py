import os
import logging
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def merge_partition(partition, df_taxi):
    return (partition
            .merge(right=(df_taxi
                          .copy()
                          .rename(columns={'lon': 'pickup_lon',
                                           'lat': 'pickup_lat',
                                           'area': 'pickup_area'})),
                   how='left',
                   left_on='pickup_location_id',
                   right_on='location_id')
            .merge(right=(df_taxi
                          .copy()
                          .rename(columns={'lon': 'dropoff_lon',
                                           'lat': 'dropoff_lat',
                                           'area': 'dropoff_area'})),
                   how='left',
                   left_on='pickup_location_id',
                   right_on='location_id'))


if __name__ == '__main__':
    DATA_DIR = 'gs://data-55acf5c126ac2d4fd4c09d61/data'
    data_raw_dir = os.path.join(DATA_DIR, 'raw')
    data_dir_preprocessed = os.path.join(DATA_DIR, 'preprocessed')
    path_taxi_zones_summary = os.path.join(DATA_DIR, 'misc', 'taxi_zones_summary.parquet')

    client = Client(address=os.environ['DASK_SCHEDULER_ADDRESS'])

    # Taxi zones summary
    df_taxi = (
        pd.read_parquet(path_taxi_zones_summary)
        .reset_index()
        .groupby('location_id', as_index=False)
        .agg(
            lon=('lon', 'mean'),
            lat=('lat', 'mean'),
            area=('area', 'sum')))

    df = dd.read_parquet(data_raw_dir).repartition(npartitions=12 * 6)

    df_new = (
        df
        .rename(columns={
            'VendorID': 'vendor_id',
            'tpep_pickup_datetime': 'pickup_datetime',
            'tpep_dropoff_datetime': 'dropoff_datetime',
            'PULocationID': 'pickup_location_id',
            'DOLocationID': 'dropoff_location_id',
            'trip_distance': 'trip_distance',
            'passenger_count': 'passenger_count'})
        .assign(
            target=lambda x: (x['dropoff_datetime'] - x['pickup_datetime']).dt.total_seconds(),
            time=lambda x: x['pickup_datetime'].dt.hour * 60 + x['pickup_datetime'].dt.minute,
            weekday=lambda x: x['pickup_datetime'].dt.weekday,
            month=lambda x: x['pickup_datetime'].dt.month,
            passenger_count=lambda x: x['passenger_count'].clip(upper=7))
        .map_partitions(
            merge_partition,
            df_taxi=df_taxi,
            align_dataframes=False)
    )

    df_new.to_parquet(data_dir_preprocessed, partition_on=['month'])

    client.close()
