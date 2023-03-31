import pandas as pd 
from tqdm import tqdm 
import time

trip_info = pd.read_csv("/Users/keivinisufaj/Downloads/nyc-taxi-trip-duration/train.csv")

trip_info.pickup_datetime = pd.to_datetime(trip_info.pickup_datetime)
trip_info.dropoff_datetime = pd.to_datetime(trip_info.dropoff_datetime)

trip_info.pickup_datetime = trip_info.pickup_datetime.astype('int64') // 10**9
trip_info.dropoff_datetime = trip_info.dropoff_datetime.astype('int64') // 10**9

for i in tqdm(range(100000)):
    lat1,lon1,ts1 = list(trip_info.iloc[i, [6,5,2]])
    lat2,lon2,ts2 = list(trip_info.iloc[i, [8,7,3]])
    with open("data/input/trip_%s.csv" % (i + 1), 'w') as f:
        f.write("0,%s,%s,%s\n" % (lat1, lon1, ts1))
        f.write("1,%s,%s,%s\n" % (lat2, lon2, ts2))