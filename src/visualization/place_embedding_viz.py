"""
Visualization of place-centric embeddings for grant prototype.

Creates:
1. UMAP projection of hexagon embeddings colored by ICE values
2. Geographic map of hexagons colored by embedding clusters
3. Comparison panels for publication

Author: Yuan Liao
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.cluster import KMeans
from umap import UMAP
import h3
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

# Setup paths
ROOT_dir = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_dir))

# Output directory
FIG_DIR = ROOT_dir / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# Style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150


def load_data():
    """Load embeddings and diversity data."""
    print("Loading data...")

    # Hexagon embeddings
    df_hex = pd.read_parquet(f'{ROOT_dir}/dbs/embeddings/place_centric_hexagon.parquet')

    # Co-presence diversity
    df_div = pd.read_parquet(f'{ROOT_dir}/dbs/cities/stockholm_copresence_diversity.parquet')

    # Aggregate diversity to hexagon level (mean across time bins)
    df_div_agg = df_div.groupby('h3_id').agg({
        'ice_birth': 'mean',
        'ice_income': 'mean',
        'n_visitors': 'sum',
        'prop_sweden': 'mean',
        'prop_other': 'mean',
        'prop_q1': 'mean',
        'prop_q4': 'mean'
    }).reset_index()

    # Merge
    df = pd.merge(df_hex, df_div_agg, on='h3_id', how='inner')

    # Get hexagon centroids for geographic plotting
    df['lat'] = df['h3_id'].apply(lambda x: h3.cell_to_latlng(x)[0])
    df['lng'] = df['h3_id'].apply(lambda x: h3.cell_to_latlng(x)[1])

    # Project to EPSG 3006 (SWEREF99 TM) for Swedish maps
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
    df['easting'], df['northing'] = transformer.transform(df['lng'].values, df['lat'].values)

    print(f"Loaded {len(df):,} hexagons with embeddings and diversity metrics")

    return df


def compute_umap(df, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """Compute UMAP projection of embeddings."""
    print("Computing UMAP projection...")

    emb_cols = [c for c in df.columns if c.startswith('x')]
    X = df[emb_cols].values

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=-1
    )

    embedding_2d = reducer.fit_transform(X)

    df['umap_1'] = embedding_2d[:, 0]
    df['umap_2'] = embedding_2d[:, 1]

    print("UMAP projection complete")
    return df


def compute_clusters(df, n_clusters=5):
    """Compute KMeans clusters on embeddings."""
    print(f"Computing {n_clusters} clusters...")

    emb_cols = [c for c in df.columns if c.startswith('x')]
    X = df[emb_cols].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    # Compute cluster statistics
    cluster_stats = df.groupby('cluster').agg({
        'ice_birth': 'mean',
        'ice_income': 'mean',
        'n_visitors': 'sum'
    }).round(3)

    print("\nCluster statistics:")
    print(cluster_stats)

    return df


def plot_umap_ice(df, save_path=None):
    """Plot UMAP colored by ICE values."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ICE Birth
    ax = axes[0]
    scatter = ax.scatter(
        df['umap_1'], df['umap_2'],
        c=df['ice_birth'],
        cmap='RdBu_r',  # Red = more Swedish, Blue = more foreign
        s=8,
        alpha=0.7,
        vmin=-0.5, vmax=0.5
    )
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Place Embeddings colored by ICE (Birth Background)\nRed=Swedish-dominant, Blue=Foreign-dominant')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ICE Birth')

    # ICE Income
    ax = axes[1]
    scatter = ax.scatter(
        df['umap_1'], df['umap_2'],
        c=df['ice_income'],
        cmap='RdBu_r',  # Red = more Q1 (low income), Blue = more Q4 (high income)
        s=8,
        alpha=0.7,
        vmin=-0.5, vmax=0.5
    )
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Place Embeddings colored by ICE (Income)\nRed=Low-income dominant, Blue=High-income dominant')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ICE Income')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_umap_clusters(df, save_path=None):
    """Plot UMAP colored by clusters."""

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        df['umap_1'], df['umap_2'],
        c=df['cluster'],
        cmap='Set1',
        s=10,
        alpha=0.7
    )
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Place Embeddings - KMeans Clusters')

    # Add legend
    handles, labels = scatter.legend_elements()
    ax.legend(handles, [f'Cluster {i}' for i in range(len(handles))],
              loc='upper right', title='Cluster')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_geographic_ice(df, save_path=None):
    """Plot geographic map of hexagons colored by ICE."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ICE Birth - Geographic
    ax = axes[0]
    scatter = ax.scatter(
        df['lng'], df['lat'],
        c=df['ice_birth'],
        cmap='RdBu_r',
        s=5,
        alpha=0.7,
        vmin=-0.5, vmax=0.5
    )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Stockholm: ICE Birth Background\n(from co-presence diversity)')
    ax.set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('ICE Birth')

    # ICE Income - Geographic
    ax = axes[1]
    scatter = ax.scatter(
        df['lng'], df['lat'],
        c=df['ice_income'],
        cmap='RdBu_r',
        s=5,
        alpha=0.7,
        vmin=-0.5, vmax=0.5
    )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Stockholm: ICE Income\n(from co-presence diversity)')
    ax.set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('ICE Income')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_geographic_clusters(df, save_path=None):
    """Plot geographic map of hexagons colored by embedding clusters."""

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        df['lng'], df['lat'],
        c=df['cluster'],
        cmap='Set1',
        s=8,
        alpha=0.7
    )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Stockholm: Places by Embedding Cluster\n(derived from mobility + POI patterns)')
    ax.set_aspect('equal')

    handles, labels = scatter.legend_elements()
    ax.legend(handles, [f'Cluster {i}' for i in range(len(handles))],
              loc='upper right', title='Cluster')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_combined_panel(df, save_path=None):
    """Create a combined panel figure for grant application (Nature style)."""

    fig = plt.figure(figsize=(14, 10))
    # Add extra column for shared colorbar
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.05],
                  hspace=0.3, wspace=0.3)

    # Panel A: UMAP colored by ICE Birth
    ax_a = fig.add_subplot(gs[0, 0])
    scatter_a = ax_a.scatter(
        df['umap_1'], df['umap_2'],
        c=df['ice_birth'],
        cmap='RdBu_r',
        s=6, alpha=0.7,
        vmin=-0.5, vmax=0.5
    )
    ax_a.set_title('(A) Embedding Space\nColored by ICE Birth')
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    # Panel B: UMAP colored by ICE Income
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(
        df['umap_1'], df['umap_2'],
        c=df['ice_income'],
        cmap='RdBu_r',
        s=6, alpha=0.7,
        vmin=-0.5, vmax=0.5
    )
    ax_b.set_title('(B) Embedding Space\nColored by ICE Income')
    for spine in ax_b.spines.values():
        spine.set_visible(False)

    # Panel C: UMAP colored by clusters
    ax_c = fig.add_subplot(gs[0, 2])
    scatter_c = ax_c.scatter(
        df['umap_1'], df['umap_2'],
        c=df['cluster'],
        cmap='Set1',
        s=6, alpha=0.7
    )
    ax_c.set_title('(C) Embedding Space\nKMeans Clusters')
    for spine in ax_c.spines.values():
        spine.set_visible(False)

    # Panel D: Geographic - ICE Birth (EPSG 3006, void style)
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.scatter(
        df['easting'], df['northing'],
        c=df['ice_birth'],
        cmap='RdBu_r',
        s=4, alpha=0.7,
        vmin=-0.5, vmax=0.5
    )
    ax_d.set_title('(D) Geographic Distribution\nICE Birth')
    ax_d.set_aspect('equal')
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    for spine in ax_d.spines.values():
        spine.set_visible(False)

    # Panel E: Geographic - ICE Income (EPSG 3006, void style)
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.scatter(
        df['easting'], df['northing'],
        c=df['ice_income'],
        cmap='RdBu_r',
        s=4, alpha=0.7,
        vmin=-0.5, vmax=0.5
    )
    ax_e.set_title('(E) Geographic Distribution\nICE Income')
    ax_e.set_aspect('equal')
    ax_e.set_xticks([])
    ax_e.set_yticks([])
    for spine in ax_e.spines.values():
        spine.set_visible(False)

    # Panel F: Geographic - Clusters (EPSG 3006, void style)
    ax_f = fig.add_subplot(gs[1, 2])
    scatter_f = ax_f.scatter(
        df['easting'], df['northing'],
        c=df['cluster'],
        cmap='Set1',
        s=4, alpha=0.7
    )
    ax_f.set_title('(F) Geographic Distribution\nEmbedding Clusters')
    ax_f.set_aspect('equal')
    ax_f.set_xticks([])
    ax_f.set_yticks([])
    for spine in ax_f.spines.values():
        spine.set_visible(False)

    # Shared ICE colorbar for A, B, D, E (right side, spanning both rows)
    cax_ice = fig.add_subplot(gs[:, 3])
    cbar_ice = fig.colorbar(scatter_a, cax=cax_ice)
    cbar_ice.set_label('ICE', fontsize=10)

    # Shared cluster legend for C and F (inside figure)
    handles_cluster, _ = scatter_c.legend_elements()
    n_clusters = df['cluster'].nunique()
    fig.legend(handles_cluster, [f'{i}' for i in range(n_clusters)],
               loc='center right', bbox_to_anchor=(1.08, 0.5),
               title='Cluster', fontsize=8, frameon=False)

    plt.suptitle('Place-Centric Embeddings: Mobility Patterns Reveal Socio-Spatial Segregation',
                 fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    plt.close()


def plot_cluster_profiles(df, save_path=None):
    """Plot cluster profiles showing ICE distributions."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ICE Birth by cluster
    ax = axes[0]
    cluster_order = df.groupby('cluster')['ice_birth'].mean().sort_values().index
    sns.boxplot(data=df, x='cluster', y='ice_birth', order=cluster_order, ax=ax, palette='Set1')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('ICE Birth')
    ax.set_title('ICE Birth by Embedding Cluster')

    # ICE Income by cluster
    ax = axes[1]
    cluster_order = df.groupby('cluster')['ice_income'].mean().sort_values().index
    sns.boxplot(data=df, x='cluster', y='ice_income', order=cluster_order, ax=ax, palette='Set1')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('ICE Income')
    ax.set_title('ICE Income by Embedding Cluster')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def main():
    print("=" * 70)
    print("Place Embedding Visualization for Grant Prototype")
    print("=" * 70)

    # Load data
    df = load_data()

    # Compute UMAP
    df = compute_umap(df)

    # Compute clusters
    df = compute_clusters(df, n_clusters=5)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Individual plots
    plot_umap_ice(df, save_path=FIG_DIR / 'place_embedding_umap_ice.png')
    plot_umap_clusters(df, save_path=FIG_DIR / 'place_embedding_umap_clusters.png')
    plot_geographic_ice(df, save_path=FIG_DIR / 'place_embedding_geo_ice.png')
    plot_geographic_clusters(df, save_path=FIG_DIR / 'place_embedding_geo_clusters.png')
    plot_cluster_profiles(df, save_path=FIG_DIR / 'place_embedding_cluster_profiles.png')

    # Combined panel for grant
    plot_combined_panel(df, save_path=FIG_DIR / 'place_embedding_combined_panel.png')

    # Save processed data for further analysis
    output_path = ROOT_dir / 'dbs' / 'cities' / 'stockholm_hexagon_embeddings_with_umap.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\nSaved processed data to: {output_path}")

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print(f"Figures saved to: {FIG_DIR}")
    print("=" * 70)

    return df


if __name__ == "__main__":
    df = main()
