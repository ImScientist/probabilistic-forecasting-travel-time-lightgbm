_ = """
    vendor_id	pickup_datetime	dropoff_datetime	pickup_location_id	dropoff_location_id	trip_distance	
    passenger_count	target	time	weekday	location_id_x	pickup_lon	pickup_lat	pickup_area	location_id_y	
    dropoff_lon	dropoff_lat	dropoff_area	month
"""

num_feats = [
    'trip_distance', 'time',
    'pickup_lon', 'pickup_lat', 'pickup_area',
    'dropoff_lon', 'dropoff_lat', 'dropoff_area']

cat_int_feats = [
    'passenger_count', 'vendor_id', 'weekday', 'month']
