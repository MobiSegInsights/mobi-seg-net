"""
12-copresence-diversity.py

Calculate co-presence diversity metrics at hexagon level for Stockholm.
Computes ICE (Index of Concentration at Extremes) for:
- Birth background: Swedish-born vs Others (foreign non-EU)
- Income: Q1 (lowest) vs Q4 (highest)

Output: Per hexagon, per time bin (morning/afternoon/evening x weekday/weekend)

Author: Yuan Liao
"""

import os
import sys
import pandas as pd
import numpy as np
import sqlalchemy
from pathlib import Path
from p_tqdm import p_map
from functools import partial
import multiprocessing

# Setup paths
ROOT_dir = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_dir))
from lib import workers

# Database connection
engine = sqlalchemy.create_engine(
    f"postgresql://{workers.keys_manager['database']['user']}:"
    f"{workers.keys_manager['database']['password']}@localhost:"
    f"{workers.keys_manager['database']['port']}/"
    f"{workers.keys_manager['database']['name']}?gssencmode=disable"
)

# Stockholm bounding box
STOCKHOLM_BOX = workers.stockholm_box  # (17.68, 59.12, 18.46, 59.48)

# Population shares for ICE adjustment (from mobi-seg-se)
# Birth background shares
SHARE_SWEDEN = 0.8044
SHARE_OTHER = 0.1107  # Foreign-born outside Europe
# Income shares (assuming roughly equal quartiles, but can adjust)
SHARE_Q1 = 0.25
SHARE_Q4 = 0.25

# Minimum visitors threshold
MIN_VISITORS = 5

# Number of parallel workers (adjust based on your CPU cores)
N_WORKERS = min(multiprocessing.cpu_count() - 2, 16)


def within_stockholm_vectorized(lat_series, lng_series):
    """Vectorized check if coordinates are within Stockholm bounding box."""
    return ((lat_series >= STOCKHOLM_BOX[1]) & (lat_series <= STOCKHOLM_BOX[3]) &
            (lng_series >= STOCKHOLM_BOX[0]) & (lng_series <= STOCKHOLM_BOX[2]))


def get_time_bin_vectorized(hour_series):
    """
    Vectorized assignment of time bin based on hour.
    Morning: 5-12, Afternoon: 12-18, Evening: 18-5
    """
    conditions = [
        (hour_series >= 5) & (hour_series < 12),
        (hour_series >= 12) & (hour_series < 18),
    ]
    choices = ['morning', 'afternoon']
    return np.select(conditions, choices, default='evening')


def get_day_type_vectorized(weekday_series):
    """
    Vectorized assignment of day type.
    Weekday: Mon-Fri (0-4), Weekend: Sat-Sun (5-6)
    """
    return np.where(weekday_series >= 5, 'weekend', 'weekday')


def load_building_demographics():
    """Load building-level demographic data."""
    print("Loading building demographic data...")

    # Load home_building mapping
    df_hb = pd.read_sql("SELECT device_aid, b_id FROM home_building", engine)

    # Load building demographic data
    df_bd = pd.read_sql("SELECT * FROM building_data", engine)

    # Merge
    df = pd.merge(df_hb, df_bd, on='b_id', how='left')

    # Calculate proportions for each device based on their home building
    # Birth background
    df['pop_birth'] = df[['sweden', 'nordic', 'eu', 'other']].sum(axis=1)
    df['prop_sweden'] = df['sweden'] / df['pop_birth']
    df['prop_other'] = df['other'] / df['pop_birth']  # Foreign non-EU
    df['prop_middle_birth'] = (df['nordic'] + df['eu']) / df['pop_birth']

    # Income (Q1-Q4)
    df['pop_income'] = df[['Q1', 'Q2', 'Q3', 'Q4']].sum(axis=1)
    df['prop_q1'] = df['Q1'] / df['pop_income']
    df['prop_q4'] = df['Q4'] / df['pop_income']
    df['prop_middle_income'] = (df['Q2'] + df['Q3']) / df['pop_income']

    # Handle NaN (buildings with zero population)
    for col in ['prop_sweden', 'prop_other', 'prop_middle_birth',
                'prop_q1', 'prop_q4', 'prop_middle_income']:
        df[col] = df[col].fillna(0)

    # Keep only needed columns
    df = df[['device_aid', 'prop_sweden', 'prop_other', 'prop_middle_birth',
             'prop_q1', 'prop_q4', 'prop_middle_income']]

    print(f"Loaded demographics for {len(df)} devices")
    return df


def process_batch(batch_id, demo_path=None):
    """
    Process a single batch of stops data.

    Parameters:
    - batch_id: batch number (0-49)
    - demo_path: path to demographics parquet file

    Returns: DataFrame with stops enriched with demographics and time bins
    """
    # Load demographics (cached per worker via reading from file)
    df_demo = pd.read_parquet(demo_path)

    # Load stops with time info
    cols = ['device_aid', 'h3_id', 'latitude', 'longitude',
            'localtime', 'weekday', 'home']
    df = pd.read_parquet(f'{ROOT_dir}/dbs/stops_pr/stops_pr_{batch_id}.parquet',
                         columns=cols)

    # Filter: exclude home visits first (faster)
    df = df[df['home'] != 1].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # Filter: Stockholm only (vectorized)
    stockholm_mask = within_stockholm_vectorized(df['latitude'], df['longitude'])
    df = df[stockholm_mask]

    if len(df) == 0:
        return pd.DataFrame()

    # Extract hour from localtime (vectorized)
    df['localtime'] = pd.to_datetime(df['localtime'], errors='coerce')
    df['hour'] = df['localtime'].dt.hour

    # Assign time bin and day type (vectorized)
    df['time_bin'] = get_time_bin_vectorized(df['hour'])
    df['day_type'] = get_day_type_vectorized(df['weekday'])

    # Merge with demographics
    df = pd.merge(df, df_demo, on='device_aid', how='inner')

    # Keep only needed columns
    df = df[['device_aid', 'h3_id', 'time_bin', 'day_type',
             'prop_sweden', 'prop_other', 'prop_middle_birth',
             'prop_q1', 'prop_q4', 'prop_middle_income']]

    return df


def ice_vectorized(df, ai_col, bi_col, oi_col, share_a, share_b):
    """
    Vectorized ICE calculation for a DataFrame.

    Returns: Series of ICE values
    """
    share_o = 1 - share_a - share_b

    ai_adj = df[ai_col] / share_a
    bi_adj = df[bi_col] / share_b
    oi_adj = df[oi_col] / share_o

    denominator = ai_adj + bi_adj + oi_adj
    ice_values = (ai_adj - bi_adj) / denominator
    ice_values = ice_values.replace([np.inf, -np.inf], np.nan)

    return ice_values


def aggregate_diversity(df_all):
    """
    Aggregate visitor demographics by hexagon-time bin and compute ICE metrics.
    Uses vectorized pandas operations for efficiency.

    Parameters:
    - df_all: DataFrame with all stops and demographics

    Returns: DataFrame with diversity metrics per hexagon-time bin
    """
    print("Aggregating diversity metrics (vectorized)...")

    # Step 1: Get unique visitors per hexagon-time bin
    # Drop duplicate (device, hexagon, time_bin, day_type) combinations
    # Keep first occurrence, preserving demographics
    df_unique = df_all.drop_duplicates(
        subset=['device_aid', 'h3_id', 'time_bin', 'day_type']
    )

    print(f"Unique visitor-place-time combinations: {len(df_unique):,}")

    # Step 2: Aggregate by hexagon-time bin using groupby
    agg_dict = {
        'device_aid': 'nunique',  # Count unique visitors
        'prop_sweden': 'mean',
        'prop_other': 'mean',
        'prop_middle_birth': 'mean',
        'prop_q1': 'mean',
        'prop_q4': 'mean',
        'prop_middle_income': 'mean'
    }

    df_agg = df_unique.groupby(['h3_id', 'time_bin', 'day_type']).agg(agg_dict)
    df_agg = df_agg.rename(columns={'device_aid': 'n_visitors'}).reset_index()

    print(f"Total hexagon-time groups: {len(df_agg):,}")

    # Step 3: Filter by minimum visitors
    df_agg = df_agg[df_agg['n_visitors'] >= MIN_VISITORS].copy()
    print(f"Groups with >= {MIN_VISITORS} visitors: {len(df_agg):,}")

    # Step 4: Calculate ICE metrics (vectorized)
    df_agg['ice_birth'] = ice_vectorized(
        df_agg,
        ai_col='prop_sweden',
        bi_col='prop_other',
        oi_col='prop_middle_birth',
        share_a=SHARE_SWEDEN,
        share_b=SHARE_OTHER
    )

    df_agg['ice_income'] = ice_vectorized(
        df_agg,
        ai_col='prop_q1',
        bi_col='prop_q4',
        oi_col='prop_middle_income',
        share_a=SHARE_Q1,
        share_b=SHARE_Q4
    )

    # Step 5: Select final columns
    df_result = df_agg[['h3_id', 'time_bin', 'day_type', 'n_visitors',
                        'ice_birth', 'ice_income',
                        'prop_sweden', 'prop_other', 'prop_q1', 'prop_q4']]

    return df_result


def main():
    print("=" * 60)
    print("Co-presence Diversity Calculation for Stockholm")
    print("=" * 60)
    print(f"Using {N_WORKERS} parallel workers")

    # Load demographics and save to temp file for parallel workers
    df_demo = load_building_demographics()
    demo_path = f'{ROOT_dir}/dbs/cities/temp_demographics.parquet'
    df_demo.to_parquet(demo_path, index=False)
    print(f"Saved demographics to temp file: {demo_path}")

    # Process all batches in parallel
    print("\nProcessing stops batches in parallel...")
    batch_ids = list(range(50))

    # Use partial to pass demo_path to each worker
    process_batch_with_path = partial(process_batch, demo_path=demo_path)

    # Use p_map for parallel processing with progress bar
    df_list = p_map(process_batch_with_path, batch_ids, num_cpus=N_WORKERS)

    # Filter out empty DataFrames and concatenate
    df_list = [df for df in df_list if len(df) > 0]
    df_all = pd.concat(df_list, ignore_index=True)

    print(f"\nTotal stops in Stockholm (non-home): {len(df_all):,}")
    print(f"Unique devices: {df_all['device_aid'].nunique():,}")
    print(f"Unique hexagons: {df_all['h3_id'].nunique():,}")

    # Aggregate and compute diversity
    df_diversity = aggregate_diversity(df_all)

    print(f"\nHexagon-time bins with >= {MIN_VISITORS} visitors: {len(df_diversity):,}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print("\nICE Birth Background (Swedish vs Foreign non-EU):")
    print(df_diversity['ice_birth'].describe())
    print("\nICE Income (Q1 vs Q4):")
    print(df_diversity['ice_income'].describe())

    # Save output
    output_path = f'{ROOT_dir}/dbs/cities/stockholm_copresence_diversity.parquet'
    df_diversity.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Also save as CSV for easy inspection
    csv_path = f'{ROOT_dir}/dbs/cities/stockholm_copresence_diversity.csv'
    df_diversity.to_csv(csv_path, index=False)
    print(f"Also saved as: {csv_path}")

    # Cleanup temp file
    if os.path.exists(demo_path):
        os.remove(demo_path)
        print("Cleaned up temp demographics file")

    return df_diversity


if __name__ == "__main__":
    df = main()
