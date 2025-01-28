import sys
from pathlib import Path
import os
import pandas as pd
import time
from tqdm import tqdm
import numpy as np


ROOT_dir = Path(__file__).parent.parent.parent
sys.path.append(ROOT_dir)
sys.path.insert(0, os.path.join(ROOT_dir, 'lib'))
days_num = {'01': 31, '02': 28, '03': 31, '04': 30, '05': 31, '06': 30,
            '07': 31, '08': 31, '09': 30, '10': 31, '11': 30, '12': 31}
m_list = [f"{i:02d}" for i in range(1, 13)]


def get_day_list(month=None):
    days = ["%02d" % (number,) for number in range(1, days_num[month] + 1)]
    return days


class DataPrep:
    def __init__(self):
        self.raw_data_folder = 'E:\\MAD_dbs\\raw_data_se_24'
        self.converted_data_folder = 'D:\\MAD_dbs\\raw_data_se_24\\format_parquet'
        self.data = None
        self.devices = None

    def device_logging(self, month=None):
        devices_m = []
        days_list = get_day_list(month=month)
        for day in days_list:
            df_list = []
            path = os.path.join(self.raw_data_folder, m, day)
            file_list = os.listdir(path)
            for file in file_list:
                file_path = os.path.join(path, file)
                print(f'Loading {file_path}')
                df_list.append(pd.read_csv(file_path, sep='\t', compression='gzip', usecols=['device_aid']))
            temp_ = pd.concat(df_list)
            temp_ = temp_[temp_.device_aid.apply(lambda x: isinstance(x, str))]
            devices_m.append(temp_['device_aid'].unique())
            del temp_
        devices_m = np.unique(np.concatenate(devices_m))
        devices_m = pd.DataFrame(devices_m, columns=['uid'])
        devices_m.to_parquet(f'dbs/devices/devices_{month}.parquet', index=False)

    def device_grouping(self, num_groups=100):
        file_path = os.path.join(ROOT_dir, 'dbs/devices/devices_grp.parquet')
        if os.path.isfile(file_path):
            print('Loading existing groups...')
            self.devices = pd.read_parquet(file_path)
        else:
            print('Grouping users...')
            devices = []
            for month in m_list:
                print(f'Processing {month}...')
                devices.append(pd.read_parquet(os.path.join(ROOT_dir, f'dbs/devices/devices_{month}.parquet')))
            devices = pd.concat(devices)
            devices.drop_duplicates(subset=['uid'], inplace=True)
            np.random.seed(68)
            devices.loc[:, 'grp'] = np.random.randint(0, num_groups, size=len(devices))
            devices.rename(columns={'uid': 'device_aid'}, inplace=True)
            devices.to_parquet(file_path, index=False)
            self.devices = devices

    def process_data(self, selectedcols=None, month=None, day=None):
        """
        :param selectedcols: a list of column names
        :type selectedcols: list
        """
        start = time.time()
        print("Data loading...")
        path = os.path.join(self.raw_data_folder, month, day)
        file_list = os.listdir(path)
        df_list = []
        for file in file_list:
            file_path = os.path.join(path, file)
            print(f'Loading {file_path}')
            df_list.append(pd.read_csv(file_path, sep='\t', compression='gzip', usecols=selectedcols))
        temp_ = pd.concat(df_list)
        del df_list
        print('Adding group id to device_aids...')
        self.data = pd.merge(temp_, self.devices[['device_aid', 'grp']],
                         on='device_aid', how='left')
        del temp_
        end = time.time()
        print(f"Data processed in {(end - start)/60} minutes.")

    def write_out(self, month=None, day=None):
        def write_data(data=None):
            dirs = [x[0] for x in os.walk(self.converted_data_folder)]
            grp = data.name
            target_dir = os.path.join(self.converted_data_folder, 'grp_' + str(int(grp)))
            if target_dir not in dirs:
                os.makedirs(target_dir)
                print("created folder : ", target_dir)
            data.to_parquet(os.path.join(target_dir, month + '_' + day + '_.parquet'), index=False)

        print(f'Saving data...')
        tqdm.pandas()
        self.data.groupby('grp').progress_apply(lambda x: write_data(data=x))


if __name__ == '__main__':
    # Stage 1- Logging device ids
    # Stage 2- Processing files
    stage = 2
    if stage == 1:
        print('Processing .csv.gz to log all device ids:')
        data_prep = DataPrep()
        cols = ['timestamp', 'device_aid', 'latitude', 'longitude', 'location_method']
        for m in m_list:
            print(f'Processing SE 2024 - month {m}:')
            data_prep.device_logging(month=m)

    if stage == 2:
        print('Processing .csv.gz into parquet by day:')
        cols = ['timestamp', 'device_aid', 'latitude', 'longitude', 'location_method']
        data_prep = DataPrep()
        print('Prepare batches...')
        data_prep.device_grouping(num_groups=50)
        trackers = m_list.copy()
        #for item in []:
        #    trackers.remove(item)
        for m in trackers:
            print(f'Processing 2024 - month {m}:')
            start = time.time()
            days_list = get_day_list(month=m)
            # if (m == '08') & (y == 2019):
            #     del days_list[:17]
            for day in days_list:
                data_prep.process_data(selectedcols=cols, month=m, day=day)
                data_prep.write_out(month=m, day=day)
            end = time.time()
            time_elapsed = (end - start)//60    #  in minutes
            print(f"Month {m} processed in {time_elapsed} minutes.")
