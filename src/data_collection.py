"""
    Collect raw NYC taxi trip data and store it in $DST.

    $DST can be both a local directory or a GCS bucket.

    The following objects are created:

    $DST
    ├── data
        ├── raw
        │   ├── data_2016-01.parquet
        │   ├── data_2016-02.parquet
        │   ├── ...
        └── misc
            └──taxi_zones_summary.parquet
"""

import os
import logging
import tempfile
import pandas as pd
import geopandas as gpd
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def collect_data(save_dir: str, year: int):
    """ Get NYC taxi data for a particular year """

    if not save_dir.startswith('gs://'):
        os.makedirs(save_dir, exist_ok=True)

    columns = [
        'VendorID',
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
        'PULocationID',
        'DOLocationID',
        'trip_distance',
        'passenger_count']

    for month in range(1, 13):
        uri = ('https://d37ci6vzurychx.cloudfront.net/trip-data/'
               f'yellow_tripdata_{year}-{month:02d}.parquet')

        dst = os.path.join(save_dir, f'data_{year}-{month:02d}.parquet')

        logger.info(f'Store data in {dst}')
        pd.read_parquet(uri, columns=columns).to_parquet(dst)


def store_taxi_zones_summary(save_dir: str):
    """ Store a dataframe with center and area of each taxi zone """

    uri = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    dst = os.path.join(save_dir, 'taxi_zones_summary.parquet')

    logger.info(f'Fetch and preprocess data from {uri} and store it in {dst}\n'
                f'We will ignore all warnings.\n')

    with tempfile.NamedTemporaryFile('w', suffix='.zip') as f:
        urlretrieve(uri, f.name)

        (gpd
         .read_file(f.name)
         .to_crs("epsg:4326")
         .rename(columns={'LocationID': 'location_id'})
         .assign(lon=lambda x: x.geometry.centroid.x,
                 lat=lambda x: x.geometry.centroid.y,
                 area=lambda x: x.geometry.area)
         .loc[:, ['location_id', 'lon', 'lat', 'area']]
         .set_index('location_id')
         .to_parquet(dst))


if __name__ == '__main__':
    BUCKET_NAME = 'artifacts-.....'

    data_dir = f'gs://{BUCKET_NAME}/data'
    data_raw_dir = os.path.join(data_dir, 'raw')
    data_dir_misc = os.path.join(data_dir, 'misc')

    collect_data(save_dir=data_raw_dir, year=2016)
    collect_data(save_dir=data_raw_dir, year=2017)
    store_taxi_zones_summary(save_dir=data_dir_misc)
