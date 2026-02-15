import pandas as pd
import sys
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm
import sqlalchemy
from howde import HoWDe_labelling
from pyspark.sql.functions import col
import pyarrow.parquet as pq
import pyarrow as pa
import matplotlib.pyplot as plt
from collections import Counter
import pickle

ROOT_dir = Path(__file__).parent.parent.parent
sys.path.append(ROOT_dir)
sys.path.insert(0, os.path.join(ROOT_dir, 'lib'))
print(ROOT_dir)

import workers as workers

# Pyspark set up
os.environ['JAVA_HOME'] = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.27.6-hotspot"
from pyspark.sql import SparkSession
import sys
from pyspark import SparkConf
# Set up pyspark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# Create new context
# Create new context
mem="50g"
n_workers = 18
spark = SparkSession.builder.config("spark.sql.files.ignoreCorruptFiles","true")\
                                            .config("spark.driver.memory", mem) \
                                            .config("spark.driver.maxResultSize", "40g") \
                                            .config("spark.executer.memory", "40g") \
                                            .config("spark.sql.session.timeZone","Europe/Stockholm")\
                                            .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                                            .config("spark.kryoserializer.buffer.max", "128m")\
                                            .config("spark.storage.memoryFraction", "0.5")\
                                            .config("spark.sql.broadcastTimeout", "7200")\
                                            .master(f"local[{n_workers}]").getOrCreate()
java_version = spark._jvm.System.getProperty("java.version")
print(f"Java version used by PySpark: {java_version}")
print('Web UI:', spark.sparkContext.uiWebUrl)

# Data location
user = workers.keys_manager['database']['user']
password = workers.keys_manager['database']['password']
port = workers.keys_manager['database']['port']
db_name = workers.keys_manager['database']['name']
engine = sqlalchemy.create_engine(f'postgresql://{user}:{password}@localhost:{port}/{db_name}?gssencmode=disable')


def work_extract(data):
    if len(data[data['location_type'] == 'W']) > 0:
        w_list = data.loc[data['location_type'] == 'W', 'loc'].tolist()
        counter = Counter(w_list)
        most_common_element, count = counter.most_common(1)[0]
    else:
        most_common_element, count = 0, 0
    return pd.Series(dict(workplace_loc=most_common_element, workplace_count=count))

if __name__ == '__main__':
    # Run HoWDe labelling
    with open(os.path.join(ROOT_dir, 'dbs/uuid_r_dict.pkl'), 'rb') as f:
        uuid_r_dict = pickle.load(f)
    df_w_list = []
    for i in range(1, 201):
        file_path = os.path.join(ROOT_dir, f'dbs/temp/stops_pr_{i}_millis.parquet')
        print(f'File {i} is being processed...')
        input_data = spark.read.parquet(file_path) \
            .select("useruuid", "loc", "start", "end")
        labeled_data = HoWDe_labelling(
            input_data=input_data,
            spark=spark,
            range_window_home=28,
            range_window_work=42,
            dhn=3,
            dn_H=0.4,
            dn_W=0.8,
            hf_H=0.2,
            hf_W=0.2,
            df_W=0.2,
            stops_output=True,
            verbose=False
        )

        # Save the results# Show the results
        df_l = labeled_data.toPandas()
        df_l = df_l.loc[df_l['location_type'] == 'W', ['useruuid', 'loc', 'location_type']]
        df_l['useruuid'] = df_l['useruuid'].map(uuid_r_dict)
        tqdm.pandas()
        df_l = df_l.groupby('useruuid').progress_apply(work_extract).reset_index()
        print(f"No. of commuters: {df_l.shape[0]}")
        df_l.to_sql('commuter_devices', engine, schema='public', index=False,
                    if_exists='append', method='multi', chunksize=5000)
