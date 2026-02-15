"""
Diagnose place-centric embeddings to check if they make sense.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

ROOT_dir = Path(__file__).parent.parent.parent

print("=" * 70)
print("Diagnosing Place-Centric Embeddings")
print("=" * 70)

# 1. Load embeddings
print("\n1. LOADING DATA")
print("-" * 40)

df_hex = pd.read_parquet(f'{ROOT_dir}/dbs/embeddings/place_centric_hexagon.parquet')
df_ind = pd.read_parquet(f'{ROOT_dir}/dbs/embeddings/place_centric_individual.parquet')
df_poi = pd.read_parquet(f'{ROOT_dir}/dbs/embeddings/place_centric_poi_category.parquet')
df_loss = pd.read_parquet(f'{ROOT_dir}/dbs/embeddings/place_centric_loss.parquet')
df_cats = pd.read_csv(f'{ROOT_dir}/dbs/cities/stockholm_poi_categories.csv')

print(f"Hexagon embeddings: {len(df_hex):,} hexagons x 64D")
print(f"Individual embeddings: {len(df_ind):,} individuals x 64D")
print(f"POI category embeddings: {len(df_poi):,} categories x 64D")
print(f"POI categories used: {len(df_cats)}")

# 2. Check loss convergence
print("\n2. LOSS CONVERGENCE")
print("-" * 40)

print(f"Total training steps: {len(df_loss)}")
print(f"Initial loss: {df_loss['loss'].iloc[0]:.4f}")
print(f"Final loss: {df_loss['loss'].iloc[-1]:.4f}")
print(f"Loss reduction: {(1 - df_loss['loss'].iloc[-1]/df_loss['loss'].iloc[0])*100:.1f}%")

# 3. Embedding statistics
print("\n3. EMBEDDING STATISTICS")
print("-" * 40)

emb_cols = [c for c in df_hex.columns if c.startswith('x')]

for name, df in [("Hexagon", df_hex), ("Individual", df_ind), ("POI", df_poi)]:
    emb = df[emb_cols].values
    print(f"\n{name} embeddings:")
    print(f"  Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")
    print(f"  Min: {emb.min():.4f}, Max: {emb.max():.4f}")
    print(f"  L2 norm (mean): {np.linalg.norm(emb, axis=1).mean():.4f}")

    # Check for degenerate embeddings (all same)
    variance_per_dim = emb.var(axis=0)
    low_var_dims = (variance_per_dim < 0.001).sum()
    print(f"  Low variance dimensions (<0.001): {low_var_dims}/64")

# 4. Embedding diversity (are embeddings distinct?)
print("\n4. EMBEDDING DIVERSITY")
print("-" * 40)

hex_emb = df_hex[emb_cols].values
n_sample = min(1000, len(hex_emb))
idx = np.random.choice(len(hex_emb), n_sample, replace=False)
sample_emb = hex_emb[idx]

# Pairwise cosine similarity
cos_sim = cosine_similarity(sample_emb)
np.fill_diagonal(cos_sim, np.nan)  # Exclude self-similarity

print(f"Hexagon pairwise cosine similarity (sample of {n_sample}):")
print(f"  Mean: {np.nanmean(cos_sim):.4f}")
print(f"  Std: {np.nanstd(cos_sim):.4f}")
print(f"  Min: {np.nanmin(cos_sim):.4f}, Max: {np.nanmax(cos_sim):.4f}")

if np.nanmean(cos_sim) > 0.95:
    print("  WARNING: Very high similarity - embeddings may be collapsed!")
elif np.nanmean(cos_sim) < 0.1:
    print("  GOOD: Embeddings show diversity")
else:
    print("  OK: Moderate similarity")

# 5. POI categories
print("\n5. POI CATEGORIES")
print("-" * 40)

print(f"Total categories: {len(df_cats)}")
print("\nSample categories:")
for cat in df_cats['category'].head(20):
    print(f"  - {cat}")

# 6. Validate against co-presence diversity
print("\n6. VALIDATION AGAINST CO-PRESENCE DIVERSITY")
print("-" * 40)

diversity_path = f'{ROOT_dir}/dbs/cities/stockholm_copresence_diversity.parquet'
try:
    df_div = pd.read_parquet(diversity_path)

    # Merge hexagon embeddings with diversity metrics
    # Aggregate diversity to hexagon level (mean across time bins)
    df_div_hex = df_div.groupby('h3_id').agg({
        'ice_birth': 'mean',
        'ice_income': 'mean',
        'n_visitors': 'sum'
    }).reset_index()

    df_merged = pd.merge(df_hex, df_div_hex, on='h3_id', how='inner')
    print(f"Hexagons with both embeddings and diversity: {len(df_merged):,}")

    if len(df_merged) > 100:
        # Correlation between embedding PCA and ICE
        from sklearn.decomposition import PCA

        emb_merged = df_merged[emb_cols].values
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb_merged)

        df_merged['emb_pc1'] = emb_2d[:, 0]
        df_merged['emb_pc2'] = emb_2d[:, 1]

        # Correlations
        corr_birth_pc1 = stats.pearsonr(df_merged['ice_birth'], df_merged['emb_pc1'])
        corr_birth_pc2 = stats.pearsonr(df_merged['ice_birth'], df_merged['emb_pc2'])
        corr_income_pc1 = stats.pearsonr(df_merged['ice_income'], df_merged['emb_pc1'])
        corr_income_pc2 = stats.pearsonr(df_merged['ice_income'], df_merged['emb_pc2'])

        print(f"\nCorrelation: ICE_birth vs PC1: r={corr_birth_pc1[0]:.3f}, p={corr_birth_pc1[1]:.4f}")
        print(f"Correlation: ICE_birth vs PC2: r={corr_birth_pc2[0]:.3f}, p={corr_birth_pc2[1]:.4f}")
        print(f"Correlation: ICE_income vs PC1: r={corr_income_pc1[0]:.3f}, p={corr_income_pc1[1]:.4f}")
        print(f"Correlation: ICE_income vs PC2: r={corr_income_pc2[0]:.3f}, p={corr_income_pc2[1]:.4f}")

        # Check if any correlation is significant
        all_p = [corr_birth_pc1[1], corr_birth_pc2[1], corr_income_pc1[1], corr_income_pc2[1]]
        if min(all_p) < 0.05:
            print("\n  PROMISING: Significant correlation found between embeddings and segregation!")
        else:
            print("\n  NOTE: No strong linear correlation with top 2 PCs (may need more dimensions)")

        # Also check using all dimensions with simple regression
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        X = df_merged[emb_cols].values
        y_birth = df_merged['ice_birth'].values
        y_income = df_merged['ice_income'].values

        # Remove NaN
        mask = ~(np.isnan(y_birth) | np.isnan(y_income))
        X = X[mask]
        y_birth = y_birth[mask]
        y_income = y_income[mask]

        if len(X) > 50:
            ridge = Ridge(alpha=1.0)
            scores_birth = cross_val_score(ridge, X, y_birth, cv=5, scoring='r2')
            scores_income = cross_val_score(ridge, X, y_income, cv=5, scoring='r2')

            print(f"\nRidge regression R² (5-fold CV):")
            print(f"  ICE_birth: {scores_birth.mean():.3f} ± {scores_birth.std():.3f}")
            print(f"  ICE_income: {scores_income.mean():.3f} ± {scores_income.std():.3f}")

            if scores_birth.mean() > 0.1 or scores_income.mean() > 0.1:
                print("\n  GOOD: Embeddings capture some segregation signal!")
            else:
                print("\n  NOTE: Low predictive power - may need tuning or more data")

except FileNotFoundError:
    print("Co-presence diversity file not found - skipping validation")

# 7. Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

issues = []
if df_loss['loss'].iloc[-1] > df_loss['loss'].iloc[0] * 0.9:
    issues.append("Loss did not converge well")
if np.nanmean(cos_sim) > 0.9:
    issues.append("Embeddings may be collapsed (too similar)")
if len(df_hex) < 100:
    issues.append("Very few hexagons")

if issues:
    print("Potential issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("No major issues detected!")
    print("Embeddings appear reasonable for prototype analysis.")
