# Module 5: Unsupervised Learning

## Introduction

Today marks a significant shift in how we think about machine learning. In Modules 2 through 4, we always had a target variable—sales, churn, fraud. We had labels, and we trained models to predict those labels.

Now we throw that away. No labels. No target variable. Unsupervised learning is about discovering structure in data when you don't know what you're looking for. You're exploring, not predicting.

This might sound less useful, but unsupervised learning solves critical business problems: customer segmentation, anomaly detection, data visualization, feature extraction. These are problems where labels don't exist or are too expensive to obtain.

When validating unsupervised learning, "right" is about usefulness, not correctness. Use internal metrics (silhouette, inertia), check stability across runs, and—most importantly—validate with domain experts. Do clusters suggest actionable strategies? A "statistically optimal" 7-cluster solution that marketing cannot operationalize is less useful than a 3-cluster solution they can act on.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** the difference between supervised and unsupervised learning
2. **Apply** K-means and DBSCAN clustering algorithms and interpret results
3. **Determine** optimal number of clusters using elbow method and silhouette scores
4. **Apply** PCA for dimensionality reduction and interpret principal components
5. **Use** manifold learning techniques (t-SNE, UMAP, PacMAP, LocalMAP) for visualization
6. **Identify** business applications for clustering and dimensionality reduction

---

## 5.1 Clustering

Clustering is the primary unsupervised learning technique for grouping observations based on similarity rather than labeled outcomes. This section covers K-means, DBSCAN, and hierarchical clustering, along with the metrics used to evaluate cluster quality.

### Supervised vs Unsupervised

The following table highlights the core differences between supervised and unsupervised learning.

| Supervised | Unsupervised |
|------------|--------------|
| Have labels | No labels |
| Learn to predict | Discover structure |
| Regression, Classification | Clustering, Dim. Reduction |

In supervised learning, the framing is "here are the right answers; learn to predict them." In unsupervised learning, the framing is "here's the data; find interesting patterns."

### Feature Scaling for Clustering

Distance-based methods like K-means, hierarchical clustering, and DBSCAN are sensitive to feature scales. A feature measured in thousands (e.g., income) will dominate distance calculations over a feature measured in single digits (e.g., number of children). Always standardize features before clustering. This connects directly to the preprocessing discussion in Module 1 -- the same `StandardScaler` applies here.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Three Components: K-Means

Even unsupervised algorithms fit our three-component framework:

| Component | K-Means |
|-----------|---------|
| **Decision Model** | Cluster assignments — each point belongs to nearest centroid |
| **Quality Measure** | Within-cluster sum of squares (inertia) |
| **Update Method** | Iterative assignment-update — alternate between assigning and moving centroids |

The key difference from supervised learning is that, without labels, we define "quality" differently. Instead of prediction error, we measure how compact and well-separated clusters are.

Distinguishing real structure from noise is important because clustering algorithms will always find clusters—even in random data. Use the gap statistic (compares quality to random data), stability analysis (cluster on subsets—real structure is stable), and multiple algorithms (if K-means, DBSCAN, and hierarchical all find similar groups, structure is more credible). Always verify clusters predict something meaningful.

### Clustering Applications

Clustering appears across many domains.

**Customer segmentation** groups customers by behavior or demographics to drive differentiated marketing. For example, a retailer might cluster customers on recency, frequency, and purchase value (RFM features) and find three natural groups: high-value loyalists who respond to early-access offers, price-sensitive occasional buyers who respond to discounts, and lapsed customers who need win-back campaigns. The business value is not the clusters themselves — it is the distinct action each segment enables.

**Anomaly detection** works because outliers are observations that do not fit any cluster well. A network intrusion system clusters normal traffic patterns; new traffic with low cluster membership or high distance to its nearest centroid signals something unusual. DBSCAN is particularly well-suited here because it explicitly labels low-density observations as noise rather than forcing them into a cluster.

Other applications include document grouping (organizing news articles by topic without predefined categories), image compression (replacing each pixel's color with its cluster centroid, reducing from millions of colors to K representative ones), and genomics (grouping genes with similar activation patterns across conditions to identify co-regulated pathways).

### K-Means Algorithm

K-means is the most widely used clustering algorithm, partitioning data into K groups by iteratively assigning points to the nearest centroid.

#### The Algorithm

1. **Choose K** (number of clusters)
2. **Randomly initialize** K centroids
3. **Assign**: Each point to nearest centroid
4. **Update**: Move centroids to mean of assigned points
5. **Repeat** until centroids stop moving (scikit-learn defaults: `max_iter=300`, `tol=1e-4`—convergence is declared when centroid movement falls below the tolerance)

The objective is to minimize the total distance from points to their centroids:

$$\text{minimize } \sum_{k=1}^{K}\sum_{x \in C_k} ||x - \mu_k||^2$$

where $k$ indexes clusters from 1 to $K$, $C_k$ is the set of points assigned to cluster $k$, and $\mu_k$ is the centroid of cluster $k$ (the mean of all points in $C_k$).

#### Strengths and Weaknesses

K-means is (i) fast and scalable, working on millions of points, (ii) easy to implement and interpret, and (iii) well suited to spherical clusters. However, it requires specifying K in advance, is sensitive to initialization, and assumes spherical, similar-sized clusters.

The assumption of "spherical" clusters means that K-means assigns points to the nearest centroid using Euclidean distance, implicitly assuming clusters are ball-shaped with equal spread in all directions. K-means essentially draws Voronoi cells (straight-line boundaries)—any cluster that can't fit in a convex cell will be problematic. For non-spherical shapes, use DBSCAN (any shape), GMMs (elliptical), or spectral clustering (complex manifolds).

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Always standardize before distance-based clustering
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',    # Smart initialization
    n_init=10,           # Run 10 times, keep best
    random_state=42
)

labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_
print(f"Inertia: {kmeans.inertia_}")
```

Watching K-means converge makes the algorithm's behavior intuitive. Start with random centroids. Each iteration has two phases: (i) assignment—each point "votes" for its nearest centroid, and (ii) update—centroids move to the center of their voters. Points switch allegiance when a centroid moves closer than their current one. Convergence happens when no point switches—a stable equilibrium.

!!! example "Numerical Example: K-Means Iterations Step by Step"

    ```python
    import numpy as np

    # 6 points that form 2 natural clusters
    X = np.array([
        [1.0, 1.0], [1.5, 2.0], [1.2, 1.5],  # Cluster A
        [5.0, 5.0], [5.5, 4.5], [5.2, 5.2],  # Cluster B
    ])

    # Initialize centroids (not optimal on purpose)
    centroids = np.array([[2.0, 3.0], [4.0, 3.0]])

    # Run 2 iterations manually
    for iteration in range(2):
        # Assignment: each point to nearest centroid
        labels = []
        for point in X:
            d0 = np.sqrt(np.sum((point - centroids[0])**2))
            d1 = np.sqrt(np.sum((point - centroids[1])**2))
            labels.append(0 if d0 < d1 else 1)

        # Update: move centroids to cluster means
        labels = np.array(labels)
        for k in range(2):
            centroids[k] = X[labels == k].mean(axis=0)

        print(f"Iteration {iteration+1}: labels={labels}, "
              f"centroids={centroids.round(2)}")
    ```

    **Output:**

    ```
    Iteration 1: labels=[0 0 0 1 1 1], centroids=[[1.23 1.5 ] [5.23 4.9 ]]
    Iteration 2: labels=[0 0 0 1 1 1], centroids=[[1.23 1.5 ] [5.23 4.9 ]]
    ```

    **Interpretation:** After just one iteration, points correctly grouped and
    centroids moved to cluster centers. Iteration 2 shows convergence—assignments
    and centroids are stable. The final inertia is 1.01 (total squared distance
    from points to their centroids).

    *Source: `computations/module5_examples.py` — `demo_kmeans_iterations()`*


### Choosing K: Elbow Method

The elbow method works in three steps. First, run K-means for K = 1, 2, 3, ..., n. Next, plot inertia versus K. Finally, look for the "elbow" where adding clusters gives diminishing returns.

```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
```

### Choosing K: Silhouette Score

For each point, measure how similar it is to its own cluster vs. other clusters:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = average distance to points in same cluster
- $b(i)$ = average distance to points in nearest other cluster

#### Interpretation

The silhouette score compares two quantities: how tight the point's own cluster is ($a$, intra-cluster cohesion) versus how far away the nearest other cluster is ($b$, inter-cluster separation). When $b \gg a$, the numerator is large and positive relative to the denominator, pushing $s$ toward 1. When $a \gg b$, the numerator is negative, pushing $s$ toward -1.

- $s = 1$: The point is far from all other clusters and tightly embedded in its own — well-clustered.
- $s = 0$: The average distance to the point's own cluster equals the average distance to the nearest other cluster — the point is on a boundary and could plausibly belong to either.
- $s = -1$: The point is closer on average to a neighboring cluster than to its own — it is likely misassigned.

```python
from sklearn.metrics import silhouette_score, silhouette_samples

# Overall score
score = silhouette_score(X_scaled, labels)

# Per-sample (for diagnostics)
sample_scores = silhouette_samples(X_scaled, labels)
```

Individual silhouette scores are useful for diagnostics as well. You can find misclassified points by looking for negative scores, identify boundary cases where scores are near zero, and detect outliers with very low scores. As a business consideration, sometimes the "right" K comes from domain knowledge, not just metrics.

When elbow and silhouette disagree, remember that the elbow method (inertia) measures compactness while the silhouette score measures both compactness and separation. Adding clusters always reduces inertia but may not improve silhouette if new clusters are not well-separated. If the elbow suggests 5 and the silhouette suggests 3, clusters 4-5 might be subdividing natural groups. Look at both metrics, examine cluster profiles, consider business constraints, and check stability. There is rarely a single "correct" K.

!!! example "Numerical Example: Elbow and Silhouette Comparison"

    ```python
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score

    # Generate data with 3 true clusters
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

    print(f"{'K':>4} {'Inertia':>12} {'Silhouette':>12}")
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        print(f"{k:>4} {kmeans.inertia_:>12.1f} {sil:>12.3f}")
    ```

    **Output:**

    ```
       K      Inertia   Silhouette
       2       5763.5        0.705
       3        566.9        0.848    ← Both metrics agree!
       4        496.4        0.664
       5        427.1        0.490
       6        375.0        0.517
       7        308.2        0.358
    ```

    **Interpretation:** Both elbow (big drop from K=2 to K=3) and silhouette
    (maximum at K=3) point to K=3, matching the true structure. When metrics agree,
    you can be confident. When they disagree, examine cluster profiles and consider
    business constraints.

    *Source: `computations/module5_examples.py` — `demo_elbow_silhouette()`*


Calculating silhouette by hand illustrates how the score measures how well each point fits its cluster. For point *i*, compute a(i) as the average distance to all other points in the same cluster, compute b(i) as the average distance to points in the nearest different cluster, then s(i) = (b - a) / max(a, b). A point with b >> a is well-placed (s approximately 1); a point with a >> b is probably in the wrong cluster (s approximately -1).

!!! example "Numerical Example: Silhouette Score by Hand"

    ```python
    import numpy as np
    from sklearn.metrics import silhouette_score

    # 5 points in 2 clusters
    X = np.array([[0, 0], [1, 0], [0.5, 0.5], [5, 0], [6, 0]])
    labels = np.array([0, 0, 0, 1, 1])

    # For point 0: a(0) = avg dist to points 1,2 in same cluster
    #              b(0) = avg dist to points 3,4 in other cluster
    print("Point  a(i)   b(i)   s(i)")
    for i in range(len(X)):
        same = [j for j in range(len(X)) if labels[j] == labels[i] and j != i]
        diff = [j for j in range(len(X)) if labels[j] != labels[i]]
        a_i = np.mean([np.linalg.norm(X[i] - X[j]) for j in same])
        b_i = np.mean([np.linalg.norm(X[i] - X[j]) for j in diff])
        s_i = (b_i - a_i) / max(a_i, b_i)
        print(f"  {i}    {a_i:.2f}   {b_i:.2f}   {s_i:.3f}")

    print(f"\nAverage silhouette: {silhouette_score(X, labels):.3f}")
    ```

    **Output:**

    ```
    Point  a(i)   b(i)   s(i)
      0    0.85   5.50   0.845
      1    0.85   4.50   0.810
      2    0.71   5.03   0.859
      3    1.00   4.51   0.778
      4    1.00   5.51   0.818

    Average silhouette: 0.822
    ```

    **Interpretation:** All points have high silhouette scores (>0.7) because
    the clusters are well-separated. Points 0-2 are close together (small a)
    and far from cluster 1 (large b). The overall score of 0.82 indicates
    excellent clustering.

    *Source: `computations/module5_examples.py` — `demo_silhouette_by_hand()`*


### DBSCAN: Density-Based Clustering

K-means assumes spherical clusters. DBSCAN handles irregular shapes, different densities, and noise or outliers.

#### Core Concepts

DBSCAN classifies each point into one of three categories. A **core point** has at least `min_samples` samples in its `eps`-neighborhood, counting the point itself (so `min_samples=5` requires the point plus four neighbors in scikit-learn). A **border point** falls within `eps` of a core point but is not itself a core point. A **noise point** is neither core nor border (labeled -1).

#### Algorithm

First, find all core points. Next, connect core points within `eps` of each other transitively. Then, assign border points to the nearest core point's cluster. Everything else is noise.

#### Connecting Core Points in Detail

Two core points belong to the same cluster if they are "density-reachable." This can be direct, meaning within `eps` of each other, or transitive, meaning A connects to B and B connects to C, so A and C belong to the same cluster. This is graph traversal where core points are nodes and edges exist between points within `eps`. Each connected component becomes a cluster.

#### Choosing eps and min_samples

Both parameters require some intuition to set well.

**eps** (neighborhood radius) defines the search radius around each point. Think of it as "how far away can a neighbor be and still count?" If eps is too small, nearly all points become noise — the algorithm sees no density because it's looking at too tight a circle. If eps is too large, all points end up in one giant cluster. A practical approach: compute each point's distance to its k-th nearest neighbor (where k = min_samples), sort those distances, and plot them — the elbow in that plot is a good eps estimate.

**min_samples** (minimum neighborhood size) sets how many samples must lie in the `eps`-neighborhood — counting the point itself — for it to be considered a core point. Higher values mean you require denser regions to count as clusters — useful for noisy data but can over-fragment real structure. A reasonable starting point: set min_samples to d + 1, where d is the number of features, and increase it if you're finding too many small clusters. If parameter tuning is frustrating, try HDBSCAN (available in `sklearn.cluster` since version 1.3) — it removes the eps parameter entirely by operating across multiple density scales.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Clusters: {n_clusters}, Noise: {n_noise}")
```

A social network analogy makes DBSCAN intuitive. Think of data points as people at a party. **Core points** are "popular" people with at least `min_samples` friends within arm's reach (`eps`). **Border points** are "acquaintances"—not popular themselves, but friends with at least one popular person. **Noise points** are "wallflowers" standing alone, not connected to any group. A cluster forms when popular people introduce each other: if Alice knows Bob and Bob knows Carol, they're all in the same social circle—even if Alice and Carol never met directly.

!!! example "Numerical Example: DBSCAN Core, Border, and Noise"

    ```python
    import numpy as np
    from sklearn.cluster import DBSCAN

    # Create 2 clusters + 3 outliers
    np.random.seed(42)
    cluster1 = np.random.randn(15, 2) * 0.5 + [0, 0]
    cluster2 = np.random.randn(15, 2) * 0.5 + [4, 0]
    outliers = np.array([[2, 3], [-3, 2], [7, -2]])
    X = np.vstack([cluster1, cluster2, outliers])

    dbscan = DBSCAN(eps=1.0, min_samples=5)
    labels = dbscan.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_core = len(dbscan.core_sample_indices_)
    n_noise = list(labels).count(-1)
    n_border = len(X) - n_core - n_noise

    print(f"Clusters found: {n_clusters}")
    print(f"Core points: {n_core}")
    print(f"Border points: {n_border}")
    print(f"Noise points: {n_noise}")
    ```

    **Output:**

    ```
    Clusters found: 2
    Core points: 29
    Border points: 1
    Noise points: 3
    ```

    **Interpretation:** DBSCAN found 2 clusters automatically (no K specified!).
    The 3 outliers we planted were correctly identified as noise (label=-1).
    Core points have ≥5 neighbors within eps=1.0; border points are on cluster edges.

    *Source: `computations/module5_examples.py` — `demo_dbscan_classification()`*


### K-Means vs DBSCAN

The following table summarizes the key tradeoffs between K-means and DBSCAN.

| Aspect | K-Means | DBSCAN |
|--------|---------|--------|
| # Clusters | Must specify | Auto-detected |
| Shapes | Spherical | Arbitrary |
| Handles noise | No | Yes (labels -1) |
| Speed | Very fast | Slower |

### Hierarchical Clustering

Hierarchical clustering builds a tree of nested clusters, revealing structure at multiple levels of granularity.

#### Agglomerative (Bottom-Up)

The agglomerative approach starts with each point as its own cluster. At each step, it finds the two closest clusters and merges them. This process repeats until one cluster remains.

#### Linkage Methods

Linkage methods define how distance between clusters is measured when deciding which two clusters to merge next.

- **Single linkage** uses the minimum distance between any two points across the two clusters — the distance between the closest pair. This is prone to the "chaining" problem described below.
- **Complete linkage** uses the maximum distance between any two points — the distance between the farthest pair. It produces compact, well-separated clusters but can be sensitive to outliers.
- **Average linkage** uses the average distance between all pairs of points, one from each cluster. It is less sensitive to outliers than complete linkage and resists chaining better than single linkage.
- **Ward linkage** merges the pair of clusters that minimizes the total within-cluster variance after merging. It tends to produce balanced, similarly-sized clusters and is the default in most packages.

The trade-off is roughly: single linkage is most flexible about cluster shape but chains badly; complete linkage is most conservative about cluster separation but can be thrown off by outliers; Ward linkage is the most reliable general-purpose choice for compact, interpretable groups.

Beware the chaining effect: single linkage can create long "chains" that connect distant clusters through a series of close pairs. Imagine two dense clusters with one stray point halfway between them. Single linkage might merge both clusters through that bridge point, even though the clusters themselves are far apart. **Ward linkage** is usually the safest choice when you want compact, interpretable groupings.

A dendrogram is a tree showing the merge history, where the Y-axis represents the distance at which clusters merged. You can cut the dendrogram at any height to obtain a corresponding number of clusters.

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# Cut to get 3 clusters
labels = fcluster(Z, t=3, criterion='maxclust')
```

### Common Misconceptions

Several common misconceptions arise when working with clustering methods.

| Misconception | Reality |
|--------------|---------|
| "There's one correct number of clusters" | Clustering is exploratory. Multiple valid solutions exist. Business context matters. |
| "K-means always works" | Fails on complex shapes, varying densities, outliers. |
| "Silhouette = 0.9 means perfect clustering" | Silhouette measures separation, not business meaning. |
| "More clusters is always better" | Reduces variance but may not be useful. Aim for interpretable segments. |

---

## 5.2 Dimensionality Reduction

As datasets grow in the number of features, several practical and statistical challenges emerge. This section covers techniques for reducing the number of features while preserving the most important structure in the data, starting with the motivation and then moving through PCA, t-SNE, UMAP, PacMAP, and LocalMAP.

### Why Reduce Dimensions?

#### The Curse of Dimensionality

High-dimensional spaces are sparse, distance metrics become less meaningful, and models overfit more easily. These challenges motivate dimensionality reduction.

#### Benefits of Dimensionality Reduction

Dimensionality reduction provides several benefits. You gain the ability to visualize data—you cannot plot 50 dimensions, but you can plot 2. Reducing dimensions also removes uninformative noise, speeds up model training with fewer features, and enables feature extraction by creating meaningful composite variables.

You are losing information when you reduce dimensions—the question is whether you are losing signal or noise. If 10 components capture 95% of variance, the last 40 combined contribute 5% (mostly noise). Verify by comparing model performance with and without reduction. That said, rare but important patterns may have low variance, and PCA does not know your target, so captured variance might not be predictive.

### Principal Component Analysis (PCA)

PCA finds new axes called principal components that are (i) linear combinations of original features, (ii) oriented to capture maximum variance, and (iii) orthogonal to one another (uncorrelated).

#### Algorithm

PCA first centers the data by subtracting the mean. It then finds the direction of maximum variance, which becomes PC1. Next, it finds the direction of maximum remaining variance perpendicular to PC1, which becomes PC2. This process continues for each subsequent component.

PCA can be understood as a rotation. Imagine your data forms an elongated cloud tilted at 45°. The original X and Y axes don't align with the cloud's natural shape. PCA rotates the coordinate system so PC1 runs along the cloud's longest axis (maximum spread) and PC2 runs perpendicular (maximum remaining spread). You haven't changed the data—just how you describe it. Now most information is concentrated in the first few axes.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Variance explained
print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Total: {sum(pca.explained_variance_ratio_):.2%}")
```

### Choosing Number of Components

Three common approaches help you decide. A **scree plot** shows variance explained versus component number, letting you look for an elbow. The **cumulative variance** approach keeps enough components to capture 80-95% of total variance. The **Kaiser criterion** retains components with eigenvalue greater than 1, a rule that applies when PCA is performed on standardized data (i.e., the correlation matrix).

**Eigenvalues and variance explained** are directly connected. Each principal component has an associated eigenvalue that equals the variance of the data along that component's direction. When the data is standardized, each original feature contributes one unit of variance, so the total variance equals the number of features. An eigenvalue greater than 1 means that component captures more variance than any single original feature — which is the Kaiser criterion's intuition. The variance explained ratio is simply each eigenvalue divided by the total. The scree plot is just a plot of eigenvalues in descending order; the elbow marks the point where additional components add diminishing information.

It helps to think about variance as "information." Variance represents how much features differ across observations — their information content. If a feature is constant, it tells you nothing (zero variance). PCA finds the axes where data varies most and preserves that variability. The 95% threshold is like "keep 95% of the signal, discard 5% noise." It's a lossy compression, like JPEG — you lose some detail but preserve the recognizable structure.

```python
pca_full = PCA()
pca_full.fit(X_scaled)

# Cumulative variance plot
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
```

!!! example "Numerical Example: PCA Variance Explained"

    ```python
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Create data with 3 latent factors + noise (10 features total)
    np.random.seed(42)
    n = 200
    z1, z2, z3 = np.random.randn(3, n)  # 3 underlying factors

    X = np.column_stack([
        z1, 0.8*z1 + 0.2*z2, z2, 0.5*z2 + 0.5*z3, z3,  # Signal
        0.7*z1 + 0.3*z3, 0.4*z1 + 0.4*z2 + 0.2*z3,     # Mixed
        0.3*z1 + 0.3*z2 + 0.4*z3,                       # Mixed
        0.1*np.random.randn(n), 0.1*np.random.randn(n) # Noise
    ])

    pca = PCA()
    pca.fit(StandardScaler().fit_transform(X))

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumulative)):
        print(f"PC{i+1}: {var:5.1%} (cumulative: {cum:5.1%})")
    ```

    **Output:**

    ```
    PC1: 45.6% (cumulative: 45.6%)
    PC2: 21.5% (cumulative: 67.2%)
    PC3: 12.5% (cumulative: 79.7%)
    PC4: 10.5% (cumulative: 90.2%)
    PC5:  8.9% (cumulative: 99.1%)  ← 5 components needed for ≥95%
    PC6:  0.3% (cumulative: 99.3%)
    ...
    ```

    **Interpretation:** The first 3 components capture ~80% of variance—matching
    our 3 underlying factors. Components 6-10 capture <1% combined (the noise).
    We could reduce 10 dimensions → 5 with minimal information loss.

    *Source: `computations/module5_examples.py` — `demo_pca_variance_explained()`*


### Interpreting PCA Loadings

Loadings show how each original feature contributes to each principal component.

| Feature  | PC1 | PC2 |
|----------|-----|-----|
| Income   | 0.8 | 0.1 |
| Age      | 0.7 | -0.2 |
| Spending | 0.6 | 0.8 |

In this example, PC1 loads on Income, Age, and Spending, suggesting an "overall affluence" dimension. PC2 loads mainly on Spending, capturing a "spending tendency" dimension.

Component naming is subjective—it is interpretation, not discovery. Two analysts might name the same loadings differently ("Wealth" vs "Financial Stability" vs "Affluence Score"). Report actual loadings alongside interpretation, acknowledge subjectivity, and validate with domain experts. If you can't tell a coherent story, the component may not be meaningfully interpretable.

```python
# Loadings are in components_ (rows = components, cols = features)
loadings = pca.components_

import polars as pl
loadings_df = pl.DataFrame(
    loadings,
    schema=feature_names
).with_row_index("PC")
```

!!! example "Numerical Example: PCA Loadings Interpretation"

    ```python
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Customer data: 3 financial + 3 engagement features
    np.random.seed(42)
    n = 200
    wealth = np.random.randn(n)      # Latent factor 1
    engagement = np.random.randn(n)  # Latent factor 2

    X = np.column_stack([
        50000 + 20000*wealth,    # Income
        10000 + 8000*wealth,     # Savings
        200000 + 80000*wealth,   # Home_Value
        20 + 10*engagement,      # Transactions
        10 + 8*engagement,       # Logins
        2 + 3*engagement,        # Support_Calls
    ])
    features = ['Income', 'Savings', 'Home_Value', 'Transactions', 'Logins', 'Support_Calls']

    pca = PCA(n_components=2)
    pca.fit(StandardScaler().fit_transform(X))

    print("Feature         PC1      PC2")
    for name, l1, l2 in zip(features, pca.components_[0], pca.components_[1]):
        print(f"{name:15} {l1:7.3f}  {l2:7.3f}")
    ```

    **Output:**

    ```
    Feature         PC1      PC2
    Income           0.425   -0.390
    Savings          0.421   -0.396
    Home_Value       0.420   -0.397
    Transactions     0.390    0.425
    Logins           0.397    0.420
    Support_Calls    0.396    0.420
    ```

    **Interpretation:** PC1 loads positively on ALL features but more heavily on
    financial ones → "Overall Customer Value." PC2 contrasts financial (negative)
    with engagement (positive) → "Activity vs. Wealth." A customer high on PC2
    is very active but not wealthy; high on PC1 is valuable overall.

    *Source: `computations/module5_examples.py` — `demo_pca_loadings()`*


### PCA + K-Means Pipeline

A common unsupervised learning workflow applies PCA for dimensionality reduction before K-means clustering. This can improve results by removing noisy features and reducing the curse of dimensionality. The pipeline below ensures that scaling, dimensionality reduction, and clustering are applied consistently, and setting `n_components=0.95` automatically selects the number of components needed to explain 95% of the variance.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),  # Retain 95% of variance
    ("kmeans", KMeans(n_clusters=3, random_state=42)),
])

clusters = pipeline.fit_predict(X)
```

### t-SNE for Visualization

While PCA assumes linear relationships, t-SNE (t-distributed Stochastic Neighbor Embedding) handles non-linear structure. Before going further, it helps to understand what "non-linear" means in this context. High-dimensional data often lies on a **manifold** — a curved, lower-dimensional surface embedded in the high-dimensional space. Think of a Swiss roll: the data lives on a 2D surface that's been curled into 3D space. The true distances between points follow the surface, not straight lines through the surrounding space. PCA can only capture structure that lies along straight directions; manifold methods can follow the curvature.

t-SNE's goal is to preserve local neighborhoods in 2D: points close in high-dimensional space stay close, while points far apart can move freely. The key parameter is `perplexity` (typically 5-50), which roughly controls the expected number of neighbors considered, and you should try multiple values to assess stability.

Perplexity works like binoculars. Low perplexity (5-10) is like zooming in—t-SNE focuses on very local neighborhoods, which can fragment single clusters into multiple blobs. High perplexity (50-100) is like zooming out—t-SNE considers more neighbors, preserving global structure but potentially merging distinct clusters. The default of 30 usually balances these tradeoffs. You should try at least 3 values (e.g., 10, 30, 50). If results change dramatically, the structure may be ambiguous.

#### Critical Caveats

Several limitations apply to t-SNE. It is stochastic, so different runs give different results. Cluster sizes in the embedding are meaningless because distances are distorted. It can create false patterns in random data and is slow for large datasets. Use t-SNE only for visualization, not as a preprocessing step.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
```

!!! example "Numerical Example: t-SNE Perplexity Sensitivity"

    ```python
    from sklearn.manifold import TSNE
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score

    # 300 points in 5 clusters
    X, labels = make_blobs(n_samples=300, centers=5, cluster_std=1.0, random_state=42)

    print("Perplexity   Silhouette (2D)   Notes")
    for perp in [5, 30, 100]:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_2d = tsne.fit_transform(X)
        sil = silhouette_score(X_2d, labels)
        notes = {5: "Local focus", 30: "Balanced", 100: "Global focus"}[perp]
        print(f"    {perp:3}           {sil:.3f}        {notes}")
    ```

    **Output:**

    ```
    Perplexity   Silhouette (2D)   Notes
          5           0.620        Local focus
         30           0.775        Balanced
        100           0.719        Global focus
    ```

    **Interpretation:** Perplexity 30 gives best cluster separation in 2D.
    Low perplexity (5) fragments clusters; high perplexity (100) loses local detail.
    The "right" perplexity depends on your data—always try multiple values.

    *Source: `computations/module5_examples.py` — `demo_tsne_perplexity()`*


Unlike PCA, t-SNE has no `.transform()` method--only `fit_transform()`. New data points cannot be projected into an existing embedding without refitting the entire dataset. You should never use t-SNE coordinates as features for a classifier because distances are distorted. Use PCA for preprocessing instead.

When considering whether to trust t-SNE clusters, remember that t-SNE preserves local neighborhoods but distorts global distances, cluster sizes, and densities. To avoid being fooled: run multiple times with different seeds/perplexity, validate with clustering on the original high-D data (if K-means finds no structure there, t-SNE may be misleading), and check perplexity sensitivity. t-SNE is for visualization and hypothesis generation—always verify clusters with methods on the original data.

### UMAP

UMAP (Uniform Manifold Approximation and Projection) is often a better alternative to t-SNE. It is faster, especially for large data, and preserves global structure more faithfully. Although UMAP can be used for preprocessing (not just visualization), PCA is usually preferred for dimensionality reduction as a preprocessing step due to its interpretability and invertibility. UMAP also produces more reproducible results across runs.

UMAP has two main hyperparameters that are worth understanding before tuning:

**n_neighbors** controls how many neighboring points UMAP considers when building its representation of local structure. Smaller values (5-10) make UMAP focus on very local relationships — fine-grained clusters emerge clearly, but global layout may be distorted. Larger values (50-100) cause UMAP to look at broader neighborhoods, which better reflects global structure but may blur distinct local clusters together. The default of 15 balances the two.

**min_dist** controls how tightly points are packed in the low-dimensional embedding. A small value (0.0-0.1) allows UMAP to pack nearby points tightly, which makes cluster boundaries sharp and visually distinct. A larger value (0.5-1.0) spreads points out more, which gives a smoother, less cluttered layout but can make clusters harder to distinguish visually. For exploratory visualization, start with the defaults; adjust `n_neighbors` first if global structure looks wrong, and `min_dist` if clusters look too compressed or too spread.

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_scaled)
```

### PacMAP

PacMAP (Pairwise Controlled Manifold Approximation) is a newer manifold learning method that explicitly balances local and global structure preservation. It achieves this by optimizing three types of point pairs simultaneously: (i) **near pairs** preserve local neighborhoods, (ii) **mid-near pairs** maintain intermediate distances, and (iii) **far pairs** prevent global collapse. During training, PacMAP gradually shifts focus from global structure (far pairs) to local structure (near pairs), producing embeddings that are more faithful to the original geometry than t-SNE or UMAP alone.

Compared to t-SNE and UMAP, PacMAP tends to produce more stable results across runs and better preserves the relative positions of distinct clusters. It is also less sensitive to hyperparameter choices—the default settings work well in most cases.

```python
import pacmap

reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10)
X_pacmap = reducer.fit_transform(X_scaled)
```

### LocalMAP

LocalMAP emphasizes local structure preservation more aggressively than UMAP or PacMAP. Where UMAP balances local and global structure and PacMAP explicitly controls the balance through pair types, LocalMAP prioritizes keeping nearest neighbors intact. This makes it particularly effective when the fine-grained local relationships—such as which observations are most similar within a cluster—matter more than how clusters relate to one another.

LocalMAP is a good choice when you want to inspect within-cluster variation or when your downstream analysis depends on local neighborhoods rather than global positioning.

### Method Comparison

The following table compares the main dimensionality reduction methods.

| Method | Speed | Global Structure | Local Structure | Use For |
|--------|-------|------------------|-----------------|---------|
| PCA | Fast | Preserved | Limited | Preprocessing, visualization |
| t-SNE | Slow | Lost | Strong | Visualization only |
| UMAP | Medium | Partially preserved | Strong | Both |
| PacMAP | Medium | Well-balanced | Strong | Visualization, comparison |
| LocalMAP | Medium | Limited | Very strong | Local neighborhood analysis |

#### Rule of Thumb

Use PCA for preprocessing and quick visualization. For detailed 2D visualizations, try UMAP or PacMAP first. Use t-SNE when local cluster separation matters most. Use PacMAP when you want a balanced view of both local and global structure. Use LocalMAP when within-cluster detail is the priority.

#### Choosing a Method

A decision flowchart can guide your choice. If you need features for a downstream model, use PCA for its stability, invertibility, and speed. If you just need a 2D visualization, try UMAP or PacMAP first. When data has non-linear structure, t-SNE, UMAP, PacMAP, and LocalMAP will outperform PCA. For large datasets (more than 10K points), UMAP and PacMAP are much faster than t-SNE. If you need reproducibility, use PCA (which is deterministic) or PacMAP (which is more stable than t-SNE). If you want to compare cluster relationships faithfully, PacMAP's balanced optimization is a strong default.

!!! example "Numerical Example: PCA vs t-SNE on Structured Data"

    ```python
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.datasets import make_moons, make_blobs
    from sklearn.metrics import silhouette_score

    # Non-linear data: two interlocking half-moons
    X_moons, labels = make_moons(n_samples=300, noise=0.05, random_state=42)

    # PCA projection
    X_pca = PCA(n_components=2).fit_transform(X_moons)
    sil_pca = silhouette_score(X_pca, labels)

    # t-SNE projection
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_moons)
    sil_tsne = silhouette_score(X_tsne, labels)

    print(f"Two Moons (non-linear):")
    print(f"  PCA silhouette:   {sil_pca:.3f}  (moons overlap)")
    print(f"  t-SNE silhouette: {sil_tsne:.3f}  (moons separated)")
    ```

    **Output:**

    ```
    Two Moons (non-linear):
      PCA silhouette:   0.331  (moons overlap)
      t-SNE silhouette: 0.646  (moons separated)
    ```

    **Interpretation:** PCA's linear projection fails on the curved "two moons"
    structure—the classes overlap in 2D. t-SNE's non-linear approach separates
    them clearly. For linear cluster structures (spherical blobs), both methods
    work well. **Use PCA for preprocessing; use t-SNE/UMAP for visualization.**

    *Source: `computations/module5_examples.py` — `demo_pca_vs_tsne()`*


### MNIST Example

MNIST is a standard benchmark for dimensionality reduction because it has a known structure you can verify visually. Each image is 28×28 pixels — 784 features — but the images are not randomly scattered through that 784-dimensional space. There are only 10 digit classes, and within each class the images share a common shape. The data therefore lives on a much lower-dimensional manifold, even though the raw representation is high-dimensional.

**Why PCA struggles on MNIST.** Pixel values in handwritten digits vary in ways that aren't well-captured by linear combinations. The difference between a "1" and a "7" is not that a few pixels are brighter — it involves curved strokes, loops, and angles that change position and orientation across different writers. PCA's 2D projection of MNIST typically shows partially overlapping clouds for each digit. You can see some gross separation (e.g., "1"s form a tight cluster because they occupy a narrow vertical strip), but digits with similar overall pixel distributions — like 4 and 9, or 3, 5, and 8 — bleed into each other.

**What t-SNE and UMAP reveal.** Non-linear methods can follow the curvature of the digit manifold and produce 2D visualizations where each digit forms a distinct island. The arrangement of those islands is also informative: visually similar digits (3, 5, 8) appear closer together, while visually distinct digits (1, 0) appear far apart. Within each cluster, you can often see a gradient — e.g., different writing styles of "4" arranged by whether the top is open or closed.

**The key lesson from MNIST** is that the choice of reduction method matters when data has non-linear structure. For raw pixel data, images, and text embeddings, non-linear methods will almost always reveal cleaner structure than PCA alone. PCA remains the right choice when you need features for a downstream model or when your data is roughly linear — but for visualization of complex data, t-SNE or UMAP give much more interpretable results.

### Common Misconceptions

Several common misconceptions arise when working with dimensionality reduction methods.

| Misconception | Reality |
|--------------|---------|
| "PCA finds the most important features" | PCA finds linear combinations. Components may not correspond to individual features. |
| "t-SNE cluster sizes are meaningful" | t-SNE distorts distances. A big cluster in t-SNE might be same size as small one in reality. |
| "More components = better" | More preserves more info but may include noise. Choose based on task. |
| "Dimensionality reduction always helps ML models" | Sometimes original features are better. Compare performance. |

---

## Reflection Questions

1. You're segmenting customers for marketing. K-means suggests 5 clusters, but your team can only create 3 campaigns. What do you do?

2. Your clustering puts 95% of data in one cluster and creates 4 tiny ones. Is this a problem? What might cause this?

3. When would you choose DBSCAN over K-means? Give a business example.

4. A colleague says they found "the optimal number of clusters." Why should you be skeptical?

5. PCA on customer data shows PC1 explains 80% of variance. Should you only use PC1?

6. You run t-SNE twice and get different-looking plots. Is one wrong?

7. A colleague says "UMAP proves our data has 5 clusters." What's wrong with this statement?

---

## Practice Problems

1. Given cluster assignments, calculate silhouette score by hand for a small example

2. Interpret PCA loadings for a business dataset (name the components)

3. Choose between K-means and DBSCAN for different data scenarios

4. Explain why t-SNE shouldn't be used for preprocessing

5. For customer data with features {Income, Age, Transactions, Days_Since_Purchase}, describe what PC1 and PC2 might represent

---

## Chapter Summary

This module introduced unsupervised learning, which discovers structure in data without labels. K-means is the most widely used clustering algorithm—fast and interpretable, but limited to spherical clusters and requiring that you specify K in advance. DBSCAN complements K-means by handling arbitrary cluster shapes and explicitly identifying outliers as noise. Silhouette scores provide a way to measure cluster quality, though a high score reflects geometric separation rather than business meaning.

On the dimensionality reduction side, PCA finds linear combinations of features that maximize variance, making it the standard choice for preprocessing and quick visualization. For revealing non-linear structure, t-SNE, UMAP, PacMAP, and LocalMAP each offer different tradeoffs: PacMAP balances local and global structure preservation, while LocalMAP prioritizes fine-grained local detail.

---

## What's Next

Module 6 introduces neural networks fundamentals, covering perceptrons and multi-layer networks, activation functions, backpropagation, and the basics of deep learning. Dimensionality reduction connects directly to this topic: neural networks automatically learn compressed representations of inputs, and that capacity for learned feature extraction is one reason deep learning works so well.
