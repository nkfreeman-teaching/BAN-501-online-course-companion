"""
Module 5: Unsupervised Learning - Numerical Examples

This script demonstrates key concepts from Module 5 with concrete calculations.
Run with: pixi run python course-companion-computations/module5_examples.py

References in Course Companion:
- demo_kmeans_iterations()          -> Module 5, Section 5.1 (K-Means Algorithm)
- demo_elbow_silhouette()           -> Module 5, Section 5.1 (Choosing K)
- demo_silhouette_by_hand()         -> Module 5, Section 5.1 (Silhouette Score)
- demo_dbscan_classification()      -> Module 5, Section 5.1 (DBSCAN)
- demo_pca_variance_explained()     -> Module 5, Section 5.2 (PCA)
- demo_pca_loadings()               -> Module 5, Section 5.2 (Interpreting Loadings)
- demo_tsne_perplexity()            -> Module 5, Section 5.2 (t-SNE)
- demo_pca_vs_tsne()                -> Module 5, Section 5.2 (Method Comparison)

Last updated: 2026-01-02
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler


def demo_kmeans_iterations():
    """
    Show K-means algorithm step by step on small 2D data.
    Traces assignment and centroid updates for 3 iterations.
    """
    print("=" * 60)
    print("K-MEANS ITERATIONS STEP BY STEP")
    print("=" * 60)

    # Simple 2D data: 6 points that should form 2 clusters
    X = np.array([
        [1.0, 1.0],   # Cluster A
        [1.5, 2.0],   # Cluster A
        [1.2, 1.5],   # Cluster A
        [5.0, 5.0],   # Cluster B
        [5.5, 4.5],   # Cluster B
        [5.2, 5.2],   # Cluster B
    ])

    print("\nData points:")
    for i, point in enumerate(X):
        print(f"  Point {i}: ({point[0]:.1f}, {point[1]:.1f})")

    # Initialize centroids (deliberately not optimal)
    centroids = np.array([
        [2.0, 3.0],   # Initial centroid 1
        [4.0, 3.0],   # Initial centroid 2
    ])

    print(f"\nInitial centroids (randomly chosen):")
    print(f"  Centroid 0: ({centroids[0, 0]:.1f}, {centroids[0, 1]:.1f})")
    print(f"  Centroid 1: ({centroids[1, 0]:.1f}, {centroids[1, 1]:.1f})")

    for iteration in range(3):
        print(f"\n{'─' * 40}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'─' * 40}")

        # Assignment step: assign each point to nearest centroid
        labels = []
        print("\nAssignment step (assign to nearest centroid):")
        for i, point in enumerate(X):
            dist_0 = np.sqrt(np.sum((point - centroids[0]) ** 2))
            dist_1 = np.sqrt(np.sum((point - centroids[1]) ** 2))
            label = 0 if dist_0 < dist_1 else 1
            labels.append(label)
            print(f"  Point {i}: dist to C0={dist_0:.2f}, dist to C1={dist_1:.2f} → Cluster {label}")

        labels = np.array(labels)

        # Update step: move centroids to mean of assigned points
        print("\nUpdate step (move centroids to cluster means):")
        new_centroids = np.zeros_like(centroids)
        for k in range(2):
            cluster_points = X[labels == k]
            new_centroids[k] = cluster_points.mean(axis=0)
            print(f"  Cluster {k}: points {list(np.where(labels == k)[0])} → "
                  f"mean = ({new_centroids[k, 0]:.2f}, {new_centroids[k, 1]:.2f})")

        # Check for convergence
        movement = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids.copy()

        if movement < 0.01:
            print(f"\nConverged! Centroids stopped moving.")
            break

    print(f"\nFinal assignments: {labels}")
    print(f"Final centroids:")
    print(f"  Cluster 0: ({centroids[0, 0]:.2f}, {centroids[0, 1]:.2f})")
    print(f"  Cluster 1: ({centroids[1, 0]:.2f}, {centroids[1, 1]:.2f})")

    # Calculate final inertia
    inertia = 0
    for i, point in enumerate(X):
        inertia += np.sum((point - centroids[labels[i]]) ** 2)
    print(f"\nFinal inertia (within-cluster sum of squares): {inertia:.2f}")


def demo_elbow_silhouette():
    """
    Compare elbow method and silhouette scores for choosing K.
    Shows where they agree and where they might disagree.
    """
    print("\n" + "=" * 60)
    print("ELBOW AND SILHOUETTE COMPARISON")
    print("=" * 60)

    # Generate data with 3 true clusters
    np.random.seed(42)
    X, true_labels = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=1.0,
        random_state=42,
    )

    print(f"\nGenerated 300 points with 3 true clusters")
    print(f"\nTrying K = 2 through 7:")
    print(f"{'K':>4} {'Inertia':>12} {'Silhouette':>12} {'Notes':>20}")
    print("-" * 52)

    results = []
    for k in range(2, 8):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, labels)

        # Add notes
        notes = ""
        if k == 3:
            notes = "← True K"

        results.append({
            'k': k,
            'inertia': inertia,
            'silhouette': silhouette,
        })

        print(f"{k:>4} {inertia:>12.1f} {silhouette:>12.3f} {notes:>20}")

    # Find elbow (biggest drop in inertia)
    inertia_drops = []
    for i in range(1, len(results)):
        drop = results[i-1]['inertia'] - results[i]['inertia']
        inertia_drops.append((results[i]['k'], drop))

    # Find best silhouette
    best_silhouette_k = max(results, key=lambda x: x['silhouette'])['k']

    print(f"\nInterpretation:")
    print(f"  - Inertia always decreases as K increases (more clusters = less within-cluster distance)")
    print(f"  - Look for the 'elbow' where the drop becomes less steep")
    print(f"  - Best silhouette score at K={best_silhouette_k}")
    print(f"  - Both methods suggest K=3, matching the true structure!")


def demo_silhouette_by_hand():
    """
    Calculate silhouette score step by step for a small example.
    Shows a(i), b(i), and s(i) for each point.
    """
    print("\n" + "=" * 60)
    print("SILHOUETTE SCORE BY HAND")
    print("=" * 60)

    # Very small example: 5 points in 2 clusters
    X = np.array([
        [0.0, 0.0],   # Cluster 0
        [1.0, 0.0],   # Cluster 0
        [0.5, 0.5],   # Cluster 0
        [5.0, 0.0],   # Cluster 1
        [6.0, 0.0],   # Cluster 1
    ])
    labels = np.array([0, 0, 0, 1, 1])

    print("\nData and cluster assignments:")
    for i in range(len(X)):
        print(f"  Point {i}: ({X[i, 0]:.1f}, {X[i, 1]:.1f}) → Cluster {labels[i]}")

    print("\nCalculating silhouette for each point:")
    print(f"{'Point':>6} {'a(i)':>8} {'b(i)':>8} {'s(i)':>8}")
    print("-" * 34)

    silhouette_values = []

    for i in range(len(X)):
        # a(i) = average distance to points in SAME cluster
        same_cluster = np.where(labels == labels[i])[0]
        same_cluster = same_cluster[same_cluster != i]  # Exclude self

        if len(same_cluster) > 0:
            a_i = np.mean([np.sqrt(np.sum((X[i] - X[j]) ** 2)) for j in same_cluster])
        else:
            a_i = 0

        # b(i) = average distance to points in NEAREST other cluster
        other_cluster = np.where(labels != labels[i])[0]
        b_i = np.mean([np.sqrt(np.sum((X[i] - X[j]) ** 2)) for j in other_cluster])

        # s(i) = (b - a) / max(a, b)
        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        silhouette_values.append(s_i)

        print(f"{i:>6} {a_i:>8.2f} {b_i:>8.2f} {s_i:>8.3f}")

    avg_silhouette = np.mean(silhouette_values)
    print(f"\nAverage silhouette score: {avg_silhouette:.3f}")

    # Verify with sklearn
    sklearn_silhouette = silhouette_score(X, labels)
    print(f"Sklearn verification: {sklearn_silhouette:.3f}")

    print(f"\nInterpretation:")
    print(f"  - a(i) measures how close a point is to its own cluster")
    print(f"  - b(i) measures how close it is to the nearest OTHER cluster")
    print(f"  - s(i) near 1: well-clustered (b >> a)")
    print(f"  - s(i) near 0: on the boundary")
    print(f"  - s(i) near -1: probably in wrong cluster (a >> b)")


def demo_dbscan_classification():
    """
    Show how DBSCAN classifies points as core, border, or noise.
    """
    print("\n" + "=" * 60)
    print("DBSCAN CORE, BORDER, AND NOISE CLASSIFICATION")
    print("=" * 60)

    # Create data with clear structure and some outliers
    np.random.seed(42)

    # Two clusters
    cluster1 = np.random.randn(15, 2) * 0.5 + [0, 0]
    cluster2 = np.random.randn(15, 2) * 0.5 + [4, 0]
    # Outliers
    outliers = np.array([[2, 3], [-3, 2], [7, -2]])

    X = np.vstack([cluster1, cluster2, outliers])

    print(f"\nData: {len(X)} points (30 in clusters + 3 outliers)")
    print(f"\nUsing eps=1.0 and min_samples=5")

    dbscan = DBSCAN(
        eps=1.0,
        min_samples=5,
    )
    labels = dbscan.fit_predict(X)

    # Identify core samples
    core_mask = np.zeros(len(X), dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_core = sum(core_mask)
    n_border = len(X) - n_core - n_noise

    print(f"\nResults:")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Core points: {n_core} (have ≥5 neighbors within eps=1.0)")
    print(f"  Border points: {n_border} (within eps of core, but <5 neighbors)")
    print(f"  Noise points: {n_noise} (isolated, no cluster)")

    print(f"\nPoint classification breakdown:")
    print(f"{'Type':>10} {'Count':>8} {'Label(s)'}")
    print("-" * 35)
    print(f"{'Core':>10} {n_core:>8}   {set(labels[core_mask])}")
    print(f"{'Border':>10} {n_border:>8}   assigned to cluster")
    print(f"{'Noise':>10} {n_noise:>8}   {{-1}}")

    print(f"\nThe 3 outliers we added were correctly identified as noise (label=-1).")
    print(f"\nDBSCAN advantage: We didn't specify K=2; it found the clusters automatically!")


def demo_pca_variance_explained():
    """
    Show how PCA captures variance and how to choose number of components.
    """
    print("\n" + "=" * 60)
    print("PCA VARIANCE EXPLAINED")
    print("=" * 60)

    # Create correlated data (10 features, but only ~3 underlying dimensions)
    np.random.seed(42)
    n_samples = 200

    # Three latent factors
    z1 = np.random.randn(n_samples)
    z2 = np.random.randn(n_samples)
    z3 = np.random.randn(n_samples)

    # 10 features as combinations of latent factors + noise
    X = np.column_stack([
        z1 + 0.1 * np.random.randn(n_samples),                    # x1: mostly z1
        0.8 * z1 + 0.2 * z2 + 0.1 * np.random.randn(n_samples),   # x2: z1 + some z2
        z2 + 0.1 * np.random.randn(n_samples),                    # x3: mostly z2
        0.5 * z2 + 0.5 * z3 + 0.1 * np.random.randn(n_samples),   # x4: z2 + z3
        z3 + 0.1 * np.random.randn(n_samples),                    # x5: mostly z3
        0.7 * z1 + 0.3 * z3 + 0.1 * np.random.randn(n_samples),   # x6: z1 + z3
        0.4 * z1 + 0.4 * z2 + 0.2 * z3 + 0.1 * np.random.randn(n_samples),  # x7
        0.3 * z1 + 0.3 * z2 + 0.4 * z3 + 0.1 * np.random.randn(n_samples),  # x8
        0.1 * np.random.randn(n_samples),                          # x9: pure noise
        0.1 * np.random.randn(n_samples),                          # x10: pure noise
    ])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nData: {n_samples} samples × 10 features")
    print(f"(Data generated from 3 latent factors + 2 noise features)")

    # Fit PCA
    pca = PCA()
    pca.fit(X_scaled)

    print(f"\nVariance explained by each component:")
    print(f"{'PC':>4} {'Variance %':>12} {'Cumulative %':>14}")
    print("-" * 32)

    cumulative = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative += var
        marker = ""
        if cumulative >= 0.95 and (i == 0 or cumulative - var < 0.95):
            marker = " ← 95% threshold"
        print(f"{i+1:>4} {var:>11.1%} {cumulative:>13.1%}{marker}")

    # Find number of components for 95%
    n_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1

    print(f"\nInterpretation:")
    print(f"  - First 3 components capture {sum(pca.explained_variance_ratio_[:3]):.1%} of variance")
    print(f"  - Need {n_95} components to reach 95%")
    print(f"  - Components 9-10 capture very little (the noise features)")
    print(f"  - We can reduce 10 → 3 dimensions with minimal information loss!")


def demo_pca_loadings():
    """
    Show how to interpret PCA loadings to name components.
    """
    print("\n" + "=" * 60)
    print("PCA LOADINGS INTERPRETATION")
    print("=" * 60)

    # Create customer data with meaningful features
    np.random.seed(42)
    n_samples = 200

    # Latent factors: wealth, engagement
    wealth = np.random.randn(n_samples)
    engagement = np.random.randn(n_samples)

    # Features that combine these factors
    income = 50000 + 20000 * wealth + 2000 * np.random.randn(n_samples)
    savings = 10000 + 8000 * wealth + 1000 * np.random.randn(n_samples)
    home_value = 200000 + 80000 * wealth + 10000 * np.random.randn(n_samples)
    transactions = 20 + 10 * engagement + 2 * np.random.randn(n_samples)
    logins = 10 + 8 * engagement + 1 * np.random.randn(n_samples)
    support_calls = 2 + 3 * engagement + 0.5 * np.random.randn(n_samples)

    X = np.column_stack([income, savings, home_value, transactions, logins, support_calls])
    feature_names = ['Income', 'Savings', 'Home_Value', 'Transactions', 'Logins', 'Support_Calls']

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA with 2 components
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    print(f"\nCustomer dataset: 6 features")
    print(f"  Financial: Income, Savings, Home_Value")
    print(f"  Engagement: Transactions, Logins, Support_Calls")

    print(f"\nPCA Loadings (how features contribute to each component):")
    print(f"{'Feature':>15} {'PC1':>10} {'PC2':>10}")
    print("-" * 37)

    for i, name in enumerate(feature_names):
        loading_1 = pca.components_[0, i]
        loading_2 = pca.components_[1, i]
        print(f"{name:>15} {loading_1:>10.3f} {loading_2:>10.3f}")

    print(f"\nVariance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")

    print(f"\nInterpretation:")
    print(f"  PC1 loads heavily on Income, Savings, Home_Value")
    print(f"      → 'Financial Status' or 'Wealth'")
    print(f"  PC2 loads heavily on Transactions, Logins, Support_Calls")
    print(f"      → 'Account Activity' or 'Engagement'")
    print(f"\nNaming is subjective! Another analyst might call PC1 'Affluence'")
    print(f"or 'Economic Standing'. Always report the actual loadings.")


def demo_tsne_perplexity():
    """
    Show how t-SNE perplexity affects the visualization.
    """
    print("\n" + "=" * 60)
    print("t-SNE PERPLEXITY SENSITIVITY")
    print("=" * 60)

    # Generate clustered data
    np.random.seed(42)
    X, labels = make_blobs(
        n_samples=300,
        centers=5,
        cluster_std=1.0,
        random_state=42,
    )

    print(f"\nData: 300 points in 5 clusters")
    print(f"\nRunning t-SNE with different perplexity values...")
    print(f"(Perplexity ≈ expected number of neighbors)")

    perplexities = [5, 30, 100]

    print(f"\n{'Perplexity':>12} {'Cluster Separation':>25} {'Notes':>25}")
    print("-" * 65)

    for perp in perplexities:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=42,
        )
        X_tsne = tsne.fit_transform(X)

        # Measure separation using silhouette on 2D projection
        sil_score = silhouette_score(X_tsne, labels)

        if perp == 5:
            notes = "Very local, may fragment clusters"
        elif perp == 30:
            notes = "Balanced local/global"
        else:
            notes = "More global, clusters may merge"

        print(f"{perp:>12} {sil_score:>25.3f} {notes:>25}")

    print(f"\nInterpretation:")
    print(f"  - Low perplexity (5): Focuses on very local structure")
    print(f"    One cluster might appear as multiple blobs")
    print(f"  - Medium perplexity (30): Usually good default")
    print(f"    Balances local and global structure")
    print(f"  - High perplexity (100): Considers more neighbors")
    print(f"    May lose fine-grained local structure")
    print(f"\nAlways try multiple perplexity values!")
    print(f"If results change dramatically, the structure may be ambiguous.")


def demo_pca_vs_tsne():
    """
    Compare PCA and t-SNE on data with different structures.
    """
    print("\n" + "=" * 60)
    print("PCA VS t-SNE ON STRUCTURED DATA")
    print("=" * 60)

    np.random.seed(42)

    # Two interlocking half-moons (a NON-LINEAR 2D structure), then lifted into
    # 50 dimensions by a random nonlinear (cosine/Fourier) feature map. The lift
    # is what makes this a real dimensionality-reduction problem: PCA now has to
    # project 50-D -> 2-D, and a *linear* projection cannot untangle the curved
    # structure the way a nonlinear method can.
    X_moons_2d, labels_moons = make_moons(
        n_samples=300,
        noise=0.05,
        random_state=42,
    )
    rng = np.random.default_rng(0)
    W = rng.standard_normal((2, 50)) * 1.5
    bias = rng.uniform(0, 2 * np.pi, 50)
    X_moons = np.cos(X_moons_2d @ W + bias)  # 300 x 50 nonlinear embedding

    print(f"\nDataset: two interlocking half-moons embedded in {X_moons.shape[1]}-D")
    print(f"(a NON-LINEAR structure), reduced to 2D for visualization.")

    # PCA: a LINEAR projection 50-D -> 2-D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_moons)
    sil_pca = silhouette_score(X_pca, labels_moons)

    # t-SNE: a NON-LINEAR embedding 50-D -> 2-D
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
    )
    X_tsne = tsne.fit_transform(X_moons)
    sil_tsne = silhouette_score(X_tsne, labels_moons)

    print(f"\nCluster separation (silhouette score on the 2D result):")
    print(f"{'Method':>10} {'Silhouette':>12} {'Interpretation':>32}")
    print("-" * 57)
    print(f"{'PCA':>10} {sil_pca:>12.3f} {'Linear projection: moons overlap':>32}")
    print(f"{'t-SNE':>10} {sil_tsne:>12.3f} {'Non-linear: moons separated':>32}")

    # For comparison: blobs that ARE linearly separable, also in 50-D.
    X_blobs, labels_blobs = make_blobs(
        n_samples=300,
        centers=3,
        n_features=50,
        cluster_std=4.0,
        random_state=42,
    )

    pca_blobs = PCA(n_components=2)
    X_pca_blobs = pca_blobs.fit_transform(X_blobs)
    sil_pca_blobs = silhouette_score(X_pca_blobs, labels_blobs)

    tsne_blobs = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
    )
    X_tsne_blobs = tsne_blobs.fit_transform(X_blobs)
    sil_tsne_blobs = silhouette_score(X_tsne_blobs, labels_blobs)

    print(f"\nFor comparison, on 50-D spherical blobs (LINEAR structure):")
    print(f"{'Method':>10} {'Silhouette':>12} {'Interpretation':>32}")
    print("-" * 57)
    print(f"{'PCA':>10} {sil_pca_blobs:>12.3f} {'Projection recovers clusters':>32}")
    print(f"{'t-SNE':>10} {sil_tsne_blobs:>12.3f} {'Also works, slightly different':>32}")

    print(f"\nKey insight:")
    print(f"  - PCA: Fast, preserves global structure, best for LINEAR relationships")
    print(f"  - t-SNE: Slower, preserves local structure, handles NON-LINEAR manifolds")
    print(f"  - For preprocessing → use PCA (stable, invertible)")
    print(f"  - For visualization → try both, but t-SNE often reveals more structure")


if __name__ == "__main__":
    demo_kmeans_iterations()
    demo_elbow_silhouette()
    demo_silhouette_by_hand()
    demo_dbscan_classification()
    demo_pca_variance_explained()
    demo_pca_loadings()
    demo_tsne_perplexity()
    demo_pca_vs_tsne()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
