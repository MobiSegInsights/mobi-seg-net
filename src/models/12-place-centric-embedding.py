"""
12-place-centric-embedding.py

Place-centric graph embedding using POI primary categories.
- Filters POI categories to those visited by >= 1% of devices
- Builds heterogeneous graph: individual -> hexagon -> poi_category
- Trains MetaPath2Vec embeddings
- Saves both individual and hexagon embeddings

Author: Yuan Liao
"""

import os
import sys
import gc
import pickle
import time
from pathlib import Path
from pprint import pprint

import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import MetaPath2Vec
from torch_geometric.data import HeteroData
from tqdm import tqdm
from p_tqdm import p_map
from functools import partial

# Setup paths
ROOT_dir = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_dir))
from lib import workers

# Stockholm bounding box
STOCKHOLM_BOX = workers.stockholm_box

# Minimum category threshold: visited by at least 1% of devices
MIN_CATEGORY_DEVICE_SHARE = 0.01

# Sample rate for individuals (to reduce computation)
# Set to 1.0 to use all individuals, 0.1 for 10% sample
INDIVIDUAL_SAMPLE_RATE = 0.15  # 15% sample for prototype


def within_stockholm_vectorized(lat_series, lng_series):
    """Vectorized check if coordinates are within Stockholm bounding box."""
    return ((lat_series >= STOCKHOLM_BOX[1]) & (lat_series <= STOCKHOLM_BOX[3]) &
            (lng_series >= STOCKHOLM_BOX[0]) & (lng_series <= STOCKHOLM_BOX[2]))


def load_batch_with_poi(batch_id):
    """Load a batch of stops with POI primary categories for Stockholm."""
    cols = ['device_aid', 'h3_id', 'latitude', 'longitude', 'home', 'primary']
    df = pd.read_parquet(f'{ROOT_dir}/dbs/stops_pr/stops_pr_{batch_id}.parquet',
                         columns=cols)

    # Filter: exclude home visits
    df = df[df['home'] != 1].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # Filter: Stockholm only
    stockholm_mask = within_stockholm_vectorized(df['latitude'], df['longitude'])
    df = df[stockholm_mask]

    if len(df) == 0:
        return pd.DataFrame()

    # Keep only needed columns
    df = df[['device_aid', 'h3_id', 'primary']]

    return df


def filter_poi_categories(df_all, min_device_share=MIN_CATEGORY_DEVICE_SHARE):
    """
    Filter POI categories to those visited by at least min_device_share of devices.

    Parameters:
    - df_all: DataFrame with 'device_aid', 'h3_id', 'primary' columns
    - min_device_share: minimum share of devices that must visit a category

    Returns:
    - set of valid category names
    """
    print("\nFiltering POI categories...")

    # Explode primary categories (each stop can have multiple POIs)
    df_exploded = df_all[['device_aid', 'primary']].copy()
    df_exploded = df_exploded.dropna(subset=['primary'])
    df_exploded = df_exploded.explode('primary')
    df_exploded = df_exploded.dropna(subset=['primary'])

    # Count unique devices per category
    total_devices = df_all['device_aid'].nunique()
    min_devices = int(total_devices * min_device_share)

    print(f"Total unique devices: {total_devices:,}")
    print(f"Minimum devices per category (>= {min_device_share*100:.1f}%): {min_devices:,}")

    # Count devices per category
    category_device_counts = df_exploded.groupby('primary')['device_aid'].nunique()

    # Filter categories
    valid_categories = set(category_device_counts[category_device_counts >= min_devices].index)

    print(f"Total unique POI categories: {len(category_device_counts):,}")
    print(f"Categories meeting threshold: {len(valid_categories):,}")

    # Show top categories
    top_cats = category_device_counts.nlargest(20)
    print("\nTop 20 POI categories by device count:")
    for cat, count in top_cats.items():
        print(f"  {cat}: {count:,} devices ({count/total_devices*100:.1f}%)")

    return valid_categories


class PlaceCentricGraph:
    """
    Build a place-centric heterogeneous graph with POI categories.

    Node types:
    - individual: mobile device users
    - hexagon: H3 spatial cells
    - poi_category: POI primary categories

    Edge types:
    - individual -> visits -> hexagon
    - hexagon -> visited_by -> individual
    - hexagon -> has_poi -> poi_category
    - poi_category -> located_at -> hexagon
    """

    def __init__(self, df_stops, valid_categories):
        """
        Parameters:
        - df_stops: DataFrame with device_aid, h3_id, primary columns
        - valid_categories: set of POI category names to include
        """
        self.df_stops = df_stops
        self.valid_categories = valid_categories
        self.graph = None
        self.individuals_mapping = None
        self.h3_mapping = None
        self.poi_mapping = None

    def build_graph(self):
        """Build the heterogeneous graph."""
        print("\nBuilding place-centric graph...")

        # Create node mappings
        unique_individuals = self.df_stops['device_aid'].unique()
        unique_hexagons = self.df_stops['h3_id'].unique()
        unique_pois = sorted(self.valid_categories)

        self.individuals_mapping = {aid: idx for idx, aid in enumerate(unique_individuals)}
        self.h3_mapping = {h3: idx for idx, h3 in enumerate(unique_hexagons)}
        self.poi_mapping = {poi: idx for idx, poi in enumerate(unique_pois)}

        print(f"Nodes - Individuals: {len(self.individuals_mapping):,}, "
              f"Hexagons: {len(self.h3_mapping):,}, "
              f"POI categories: {len(self.poi_mapping):,}")

        # Build individual <-> hexagon edges
        df_ih = self.df_stops[['device_aid', 'h3_id']].drop_duplicates()
        df_ih['src_idx'] = df_ih['device_aid'].map(self.individuals_mapping)
        df_ih['dst_idx'] = df_ih['h3_id'].map(self.h3_mapping)

        # Build hexagon <-> poi_category edges
        df_hp = self.df_stops[['h3_id', 'primary']].copy()
        df_hp = df_hp.dropna(subset=['primary'])
        df_hp = df_hp.explode('primary')
        df_hp = df_hp[df_hp['primary'].isin(self.valid_categories)]
        df_hp = df_hp.drop_duplicates()
        df_hp['src_idx'] = df_hp['h3_id'].map(self.h3_mapping)
        df_hp['dst_idx'] = df_hp['primary'].map(self.poi_mapping)
        df_hp = df_hp.dropna(subset=['src_idx', 'dst_idx'])

        print(f"Edges - Individual-Hexagon: {len(df_ih):,}, "
              f"Hexagon-POI: {len(df_hp):,}")

        # Create HeteroData graph
        self.graph = HeteroData()

        # Add node indices
        self.graph['individual'].y_index = torch.tensor(
            list(range(len(self.individuals_mapping))), dtype=torch.long)
        self.graph['hexagon'].y_index = torch.tensor(
            list(range(len(self.h3_mapping))), dtype=torch.long)
        self.graph['poi_category'].y_index = torch.tensor(
            list(range(len(self.poi_mapping))), dtype=torch.long)

        # Add individual <-> hexagon edges
        edge_ih = torch.tensor(df_ih[['src_idx', 'dst_idx']].values.T, dtype=torch.long)
        self.graph['individual', 'visits', 'hexagon'].edge_index = edge_ih

        edge_hi = torch.tensor(df_ih[['dst_idx', 'src_idx']].values.T, dtype=torch.long)
        self.graph['hexagon', 'visited_by', 'individual'].edge_index = edge_hi

        # Add hexagon <-> poi_category edges
        edge_hp = torch.tensor(df_hp[['src_idx', 'dst_idx']].values.astype(int).T, dtype=torch.long)
        self.graph['hexagon', 'has_poi', 'poi_category'].edge_index = edge_hp

        edge_ph = torch.tensor(df_hp[['dst_idx', 'src_idx']].values.astype(int).T, dtype=torch.long)
        self.graph['poi_category', 'located_at', 'hexagon'].edge_index = edge_ph

        print(f"\nConstructed HeteroData: {self.graph}")

        return self.graph


class PlaceEmbeddingTrainer:
    """Train MetaPath2Vec embeddings on the place-centric graph."""

    def __init__(self, graph):
        self.graph = graph

        # Setup device
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch_geometric.is_xpu_available():
            self.device = torch.device('xpu')
        else:
            self.device = torch.device('cpu')
        print(f'Device: {self.device}')

        self.model = None
        self.optimizer = None
        self.loader = None
        self.loss_tracker = {}

    def init_model(self, embedding_dim=64, walk_length=40, context_size=7,
                   walks_per_node=100, num_negative_samples=5, batch_size=128):
        """Initialize MetaPath2Vec model."""

        # Define metapath: individual -> hexagon -> poi_category -> hexagon -> individual
        metapath = [
            ('individual', 'visits', 'hexagon'),
            ('hexagon', 'has_poi', 'poi_category'),
            ('poi_category', 'located_at', 'hexagon'),
            ('hexagon', 'visited_by', 'individual'),
        ]

        print(f"\nMetapath: {metapath}")

        self.model = MetaPath2Vec(
            self.graph.edge_index_dict,
            embedding_dim=embedding_dim,
            metapath=metapath,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            sparse=True
        ).to(self.device)

        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

        print(f"Model initialized with {embedding_dim}D embeddings")
        print("Creating data loader (generating random walks - may take a few minutes)...")

        self.loader = self.model.loader(batch_size=batch_size, shuffle=True, num_workers=0)

        print(f"Data loader ready. Total batches per epoch: {len(self.loader)}")

    def train_epoch(self, epoch, log_steps=50, patience=5, min_delta=0.0005):
        """Train for one epoch with early stopping."""
        self.model.train()
        total_loss = 0
        best_loss = float('inf')
        epochs_without_improvement = 0
        self.loss_tracker[epoch] = []

        for i, (pos_rw, neg_rw) in enumerate(self.loader):
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if (i + 1) % log_steps == 0:
                avg_loss = total_loss / log_steps
                print(f'Epoch: {epoch}, Step: {i + 1:03d}/{len(self.loader)}, '
                      f'Loss: {avg_loss:.4f}')
                self.loss_tracker[epoch].append((i + 1, avg_loss))
                total_loss = 0

                # Early stopping check
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f'Early stopping at Epoch {epoch}, Step {i + 1}')
                    return True

        return False

    def get_embeddings(self, graph_builder):
        """Extract embeddings for individuals and hexagons."""
        self.model.eval()

        # Individual embeddings
        z_individual = self.model(
            'individual',
            batch=self.graph['individual'].y_index.to(self.device)
        ).cpu().detach().numpy()

        # Hexagon embeddings
        z_hexagon = self.model(
            'hexagon',
            batch=self.graph['hexagon'].y_index.to(self.device)
        ).cpu().detach().numpy()

        # POI category embeddings (bonus)
        z_poi = self.model(
            'poi_category',
            batch=self.graph['poi_category'].y_index.to(self.device)
        ).cpu().detach().numpy()

        # Build DataFrames with reverse mappings
        i_reverse = {v: k for k, v in graph_builder.individuals_mapping.items()}
        h_reverse = {v: k for k, v in graph_builder.h3_mapping.items()}
        p_reverse = {v: k for k, v in graph_builder.poi_mapping.items()}

        # Individual DataFrame
        df_individual = pd.DataFrame(
            z_individual,
            columns=[f'x{i}' for i in range(z_individual.shape[1])]
        )
        df_individual['node_idx'] = range(len(df_individual))
        df_individual['device_aid'] = df_individual['node_idx'].map(i_reverse)

        # Hexagon DataFrame
        df_hexagon = pd.DataFrame(
            z_hexagon,
            columns=[f'x{i}' for i in range(z_hexagon.shape[1])]
        )
        df_hexagon['node_idx'] = range(len(df_hexagon))
        df_hexagon['h3_id'] = df_hexagon['node_idx'].map(h_reverse)

        # POI category DataFrame
        df_poi = pd.DataFrame(
            z_poi,
            columns=[f'x{i}' for i in range(z_poi.shape[1])]
        )
        df_poi['node_idx'] = range(len(df_poi))
        df_poi['poi_category'] = df_poi['node_idx'].map(p_reverse)

        return df_individual, df_hexagon, df_poi


def main():
    start_time = time.time()

    print("=" * 70)
    print("Place-Centric Graph Embedding with POI Categories")
    print("=" * 70)
    print(f"Individual sample rate: {INDIVIDUAL_SAMPLE_RATE*100:.0f}%")

    # Step 1: Load Stockholm stops data with POI categories
    step_start = time.time()
    print("\nStep 1: Loading Stockholm stops data...")
    batch_ids = list(range(50))
    df_list = p_map(load_batch_with_poi, batch_ids, num_cpus=8)
    df_list = [df for df in df_list if len(df) > 0]
    df_all = pd.concat(df_list, ignore_index=True)

    print(f"Total stops: {len(df_all):,}")
    print(f"Unique devices: {df_all['device_aid'].nunique():,}")
    print(f"Unique hexagons: {df_all['h3_id'].nunique():,}")

    # Sample individuals to reduce computation
    if INDIVIDUAL_SAMPLE_RATE < 1.0:
        print(f"\nSampling {INDIVIDUAL_SAMPLE_RATE*100:.0f}% of individuals for efficiency...")
        all_devices = df_all['device_aid'].unique()
        n_sample = int(len(all_devices) * INDIVIDUAL_SAMPLE_RATE)
        np.random.seed(42)  # Reproducible
        sampled_devices = np.random.choice(all_devices, size=n_sample, replace=False)
        df_all = df_all[df_all['device_aid'].isin(sampled_devices)]
        print(f"After sampling: {len(df_all):,} stops, {df_all['device_aid'].nunique():,} devices")

    print(f"Step 1 completed in {time.time() - step_start:.1f}s")

    # Step 2: Filter POI categories
    step_start = time.time()
    print("\nStep 2: Filtering POI categories...")
    valid_categories = filter_poi_categories(df_all, min_device_share=MIN_CATEGORY_DEVICE_SHARE)

    # Save valid categories for reference
    categories_path = f'{ROOT_dir}/dbs/cities/stockholm_poi_categories.csv'
    pd.DataFrame({'category': sorted(valid_categories)}).to_csv(categories_path, index=False)
    print(f"Saved {len(valid_categories)} valid categories to {categories_path}")
    print(f"Step 2 completed in {time.time() - step_start:.1f}s")

    # Step 3: Build graph
    step_start = time.time()
    print("\nStep 3: Building heterogeneous graph...")
    graph_builder = PlaceCentricGraph(df_all, valid_categories)
    graph = graph_builder.build_graph()

    # Save graph for future use
    graph_path = f'{ROOT_dir}/dbs/cities/stockholm_place_graph.pt'
    torch.save(graph, graph_path)
    print(f"Saved graph to {graph_path}")

    # Also save mappings
    mappings = {
        'individuals': graph_builder.individuals_mapping,
        'hexagons': graph_builder.h3_mapping,
        'poi_categories': graph_builder.poi_mapping
    }
    mappings_path = f'{ROOT_dir}/dbs/cities/stockholm_place_graph_mappings.pkl'
    with open(mappings_path, 'wb') as f:
        pickle.dump(mappings, f)
    print(f"Saved mappings to {mappings_path}")
    print(f"Step 3 completed in {time.time() - step_start:.1f}s")

    # Step 4: Train embeddings
    step_start = time.time()
    print("\nStep 4: Training MetaPath2Vec embeddings...")
    trainer = PlaceEmbeddingTrainer(graph)

    # Model parameters - optimized for efficiency
    params = {
        'embedding_dim': 64,
        'walk_length': 20,        # Reduced from 40
        'context_size': 7,
        'walks_per_node': 20,     # Reduced from 100
        'num_negative_samples': 5,
        'batch_size': 256         # Increased from 128
    }
    print("\nModel parameters (optimized for prototype):")
    pprint(params)

    trainer.init_model(**params)

    # Training loop
    max_epochs = 6
    for epoch in range(max_epochs):
        epoch_start = time.time()
        should_stop = trainer.train_epoch(epoch, log_steps=50, patience=5, min_delta=0.0005)
        print(f"Epoch {epoch} completed in {time.time() - epoch_start:.1f}s")
        if should_stop:
            break

    print(f"Step 4 (training) completed in {time.time() - step_start:.1f}s")

    # Step 5: Extract and save embeddings
    step_start = time.time()
    print("\nStep 5: Extracting embeddings...")
    df_individual, df_hexagon, df_poi = trainer.get_embeddings(graph_builder)

    print(f"Individual embeddings: {len(df_individual):,} x {params['embedding_dim']}D")
    print(f"Hexagon embeddings: {len(df_hexagon):,} x {params['embedding_dim']}D")
    print(f"POI category embeddings: {len(df_poi):,} x {params['embedding_dim']}D")

    # Save embeddings
    output_dir = f'{ROOT_dir}/dbs/embeddings'

    individual_path = f'{output_dir}/place_centric_individual.parquet'
    df_individual.to_parquet(individual_path, index=False)
    print(f"Saved individual embeddings to {individual_path}")

    hexagon_path = f'{output_dir}/place_centric_hexagon.parquet'
    df_hexagon.to_parquet(hexagon_path, index=False)
    print(f"Saved hexagon embeddings to {hexagon_path}")

    poi_path = f'{output_dir}/place_centric_poi_category.parquet'
    df_poi.to_parquet(poi_path, index=False)
    print(f"Saved POI category embeddings to {poi_path}")

    # Save loss history
    loss_list = [item for sublist in trainer.loss_tracker.values() for item in sublist]
    df_loss = pd.DataFrame(loss_list, columns=['step', 'loss'])
    df_loss['epoch'] = df_loss.index // len(loss_list) if loss_list else 0
    loss_path = f'{output_dir}/place_centric_loss.parquet'
    df_loss.to_parquet(loss_path, index=False)
    print(f"Saved loss history to {loss_path}")

    # Save parameters
    params_path = f'{output_dir}/place_centric_params.pkl'
    with open(params_path, 'wb') as f:
        pickle.dump(params, f)

    print(f"Step 5 completed in {time.time() - step_start:.1f}s")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"POI categories used: {len(valid_categories)}")
    print(f"Individual embeddings: {len(df_individual):,}")
    print(f"Hexagon embeddings: {len(df_hexagon):,}")
    print(f"Output directory: {output_dir}")
    print(f"Total runtime: {total_time/60:.1f} minutes")

    # Cleanup
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return df_individual, df_hexagon, df_poi


if __name__ == "__main__":
    df_i, df_h, df_p = main()
