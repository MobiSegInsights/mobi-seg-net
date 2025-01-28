import os
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
from p_tqdm import p_map
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


ROOT_dir = Path(__file__).parent.parent
with open(os.path.join(ROOT_dir, 'dbs', 'keys.yaml')) as f:
    keys_manager = yaml.load(f, Loader=yaml.FullLoader)


se_box = (11.0273686052, 55.3617373725, 23.9033785336, 69.1062472602)
stockholm_box = (17.6799476147,59.1174841345,18.4572303295,59.475092515)


def within_se_time(latitude, longitude):
    if (latitude >= se_box[1]) & (latitude <= se_box[3]):
        if (longitude >= se_box[0]) & (longitude <= se_box[2]):
            return 1
    return 0


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) in km
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


class TimeProcessing:
    def __init__(self, data=None):
        # device_aid, timestamp, latitude, longitude
        self.data = data

    def time_processing(self):
        tqdm.pandas()
        self.data.loc[:, 'datetime'] = self.data['timestamp'].progress_apply(lambda x: datetime.fromtimestamp(x))
        self.data.loc[:, 'se_time'] = self.data.\
            progress_apply(lambda row: within_se_time(row['latitude'], row['longitude']), axis=1)
        print("Share of data in Sweden time: %.2f %%" % (self.data.loc[:, 'se_time'].sum() / len(self.data) * 100))
        # Focus on Sweden, skipping functions of time_zone_parallel and convert_to_local_time
        self.data = self.data.loc[self.data['se_time'] == 1, :]
        k = 'Europe/Stockholm'
        self.data = self.data.drop(columns=['se_time'])
        self.data.loc[:, "localtime"] = self.data['datetime'].dt.tz_localize('UTC').dt.tz_convert(k)
        print('Time processed done.')


    def time_enrich(self):
        # Add start time hour and duration in minute
        self.data['localtime'] = pd.to_datetime(self.data['localtime'], errors='coerce')
        self.data.loc[:, 'hour'] = self.data.loc[:, 'localtime'].dt.hour
        self.data.loc[:, 'month'] = self.data.loc[:, 'localtime'].dt.month
        self.data.loc[:, 'year'] = self.data.loc[:, 'localtime'].dt.year
        self.data.loc[:, 'weekday'] = self.data.loc[:, 'localtime'].dt.dayofweek
        self.data.loc[:, 'week'] = self.data.loc[:, 'localtime'].dt.isocalendar().week
        self.data.loc[:, 'date'] = self.data.loc[:, 'localtime'].dt.date
        # Add individual sequence index
        self.data = self.data.sort_values(by=['device_aid', 'timestamp'], ascending=True)
        #
        # def indi_seq(df):
        #     df.loc[:, 'seq'] = range(1, len(df) + 1)
        #     return df
        #
        # reslt = p_map(indi_seq, [g for _, g in self.data.groupby('device_aid', group_keys=True)])
        # self.data = pd.concat(reslt)


class TimeProcessingStops:
    def __init__(self, data=None):
        # device_aid, start, end, latitude, longitude
        self.data = data

    def time_processing(self):
        self.data.loc[:, 'dur'] = (self.data.loc[:, 'end'] - self.data.loc[:, 'start']) / 60    # min
        print('Convert to datetime.')
        tqdm.pandas()
        self.data.loc[:, 'datetime'] = self.data['start'].progress_apply(lambda x: datetime.fromtimestamp(x))
        self.data.loc[:, 'leaving_datetime'] = self.data['end'].progress_apply(lambda x: datetime.fromtimestamp(x))
        self.data.loc[:, 'se_time'] = self.data.\
            progress_apply(lambda row: within_se_time(row['latitude'], row['longitude']), axis=1)
        print("Share of data in Sweden time: %.2f %%" % (self.data.loc[:, 'de_time'].sum() / len(self.data) * 100))

    def time_enrich(self):
        # Add start time hour and duration in minute
        for var, fx in zip(('localtime', 'leaving_localtime'), ('', 'leaving_')):
            self.data[var] = pd.to_datetime(self.data[var], errors='coerce')
            self.data.loc[:, f'{fx}hour'] = self.data.loc[:, var].dt.hour
            self.data.loc[:, f'{fx}weekday'] = self.data.loc[:, var].dt.dayofweek
            self.data.loc[:, f'{fx}week'] = self.data.loc[:, var].dt.isocalendar().week
            self.data.loc[:, f'{fx}date'] = self.data.loc[:, var].dt.date
        # Add individual sequence index
        self.data = self.data.sort_values(by=['device_aid', 'start'], ascending=True)

        def indi_seq(df):
            df.loc[:, 'seq'] = range(1, len(df) + 1)
            return df

        reslt = p_map(indi_seq, [g for _, g in self.data.groupby('device_aid', group_keys=True)])
        self.data = pd.concat(reslt)


def long_tail_distr(data=None, col_name=None, x_lb=None, y_lb=None, bin_num=None):
    print(min(data[col_name]), max(data[col_name]))
    lower, upper = min(data[col_name]), max(data[col_name])
    bins = np.logspace(lower, np.log(upper), bin_num)
    hist, edges = np.histogram(data[col_name], bins=bins, density=True)
    x = (edges[1:] + edges[:-1]) / 2.
    xx, yy = zip(*[(i, j) for (i, j) in zip(x, hist) if j > 0])
    fig, ax = plt.subplots()
    ax.plot(xx, yy, marker='.')
    plt.axvline(x=data[col_name].median(), color='r', linestyle='dashed', linewidth=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(x_lb)
    ax.set_ylabel(y_lb)
