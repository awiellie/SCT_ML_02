import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Mall_Customers.csv"     
OUT_DIR = BASE_DIR / "outputs"                  
OUT_DIR.mkdir(exist_ok=True)

LABELED_CSV = OUT_DIR / "mall_customers_with_clusters.csv"
PROFILE_CSV = OUT_DIR / "cluster_profile.csv"
SCATTER_PNG = OUT_DIR / "clusters_scatter.png"
ELBOW_PNG   = OUT_DIR / "elbow_plot.png"
SIL_PNG     = OUT_DIR / "silhouette_scores.png"


def main():
    
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]  
    if "Gender" in df.columns:
        df["Gender_num"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(float)

    # Features representing purchase behavior
    feat_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    # Fallback in case names differ
    feat_cols = [c for c in feat_cols if c in df.columns]
    if len(feat_cols) < 2:
        feat_cols = [c for c in df.columns
                     if pd.api.types.is_numeric_dtype(df[c]) and c != "CustomerID"][:2]

    X = df[feat_cols].copy()
    X_clean = X.dropna()
    df_clean = df.loc[X_clean.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean.values)

    # Model selection
    k_range = range(2, 11)
    inertias, sil_scores = [], []
    best_k, best_sil = None, -1

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)
        if sil > best_sil:
            best_sil, best_k = sil, k

    # ---------- Final fit ----------
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    df_clean["Cluster"] = labels

    # Centers 
    centers_scaled = km.cluster_centers_
    centers_unscaled = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers_unscaled, columns=feat_cols)
    centers_df["Cluster"] = range(best_k)

    # Cluster profile 
    profile = df_clean.groupby("Cluster")[feat_cols].mean().round(2)
    profile["Count"] = df_clean["Cluster"].value_counts().sort_index()
    profile.to_csv(PROFILE_CSV)

    # Save labeled dataset 
    df_out = df.copy()
    df_out.loc[df_clean.index, "Cluster"] = df_clean["Cluster"]
    df_out.to_csv(LABELED_CSV, index=False)

    # Plots 
    # Elbow
    plt.figure()
    plt.plot(list(k_range), inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (WSS)")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.savefig(ELBOW_PNG)
    plt.close()

    # Silhouette
    plt.figure()
    plt.plot(list(k_range), sil_scores, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Scores by k")
    plt.tight_layout()
    plt.savefig(SIL_PNG)
    plt.close()

    # 2D scatter 
    plt.figure()
    for c in sorted(df_clean["Cluster"].unique()):
        s = df_clean[df_clean["Cluster"] == c]
        plt.scatter(s[feat_cols[0]], s[feat_cols[1]], s=15)
    plt.scatter(centers_df[feat_cols[0]], centers_df[feat_cols[1]],
                marker="X", s=150, edgecolors="black")
    plt.xlabel(feat_cols[0])
    plt.ylabel(feat_cols[1])
    plt.title(f"K-Means Clusters (k={best_k})")
    plt.tight_layout()
    plt.savefig(SCATTER_PNG)
    plt.close()

    print(f"Features used: {feat_cols}")
    print(f"Best k by silhouette: {best_k} (score={best_sil:.3f})")
    print("Outputs saved:",
          LABELED_CSV, PROFILE_CSV, SCATTER_PNG, ELBOW_PNG, SIL_PNG, sep="\n- ")

if __name__ == "__main__":
    main()
