### Clustering Algorithms

Clustering is an **unsupervised learning** technique where we group similar data points together.

Goal:
- points in the same cluster should be similar
- points in different clusters should be different

---

### Common Use Cases

- Customer segmentation
- Grouping similar documents/news
- Image segmentation
- Anomaly detection (some clustering methods)

---

### 1) K-Means Clustering

K-Means groups data into `K` clusters using **centroids** (cluster centers).

#### Idea

- choose `K` centroids
- assign each point to nearest centroid
- update centroid (mean of assigned points)
- repeat until centroids stop changing

---

#### Steps

1. Choose number of clusters `K`
2. Initialize `K` centroids (random)
3. Assign each point to nearest centroid
4. Update centroids (mean)
5. Repeat steps 3–4 until convergence

---

#### Distance (usually Euclidean)

$$
d(x,c)=\sqrt{\sum_{j=1}^{n}(x_j-c_j)^2}
$$

---

#### Objective (What K-Means Minimizes)

It minimizes within-cluster sum of squares (WCSS / inertia):

$$
J=\sum_{k=1}^{K}\sum_{x_i\in C_k}||x_i-\mu_k||^2
$$

where:
- $\mu_k$ = centroid of cluster $k$

---

#### Choosing Best K (Elbow Method)

Try multiple `K` values and plot WCSS:

```text
K = 1,2,3,4,5...
```

Choose K at the “elbow” point (where improvement slows).

---

#### Pros / Cons

Pros:
- simple and fast
- works well for spherical/separable clusters

Cons:
- must choose `K`
- sensitive to outliers
- depends on initialization (can give different results)
- struggles with non-spherical clusters

> Feature scaling is important for K-Means.

---

### 2) Hierarchical Clustering

Hierarchical clustering builds a **tree of clusters** (dendrogram).

Two types:
- Agglomerative (bottom-up)
- Divisive (top-down)

Most common: **Agglomerative**.

---

#### Agglomerative (Bottom-Up) Steps

1. Start: each point is its own cluster
2. Merge the two closest clusters
3. Repeat until one cluster remains

Then choose number of clusters by cutting the dendrogram.

---

#### Linkage Methods (How to measure cluster distance)

- Single linkage: nearest points
- Complete linkage: farthest points
- Average linkage: average distance
- Ward linkage: minimizes variance increase (very common)

---

#### Pros / Cons

Pros:
- no need to pre-choose `K` (can decide using dendrogram)
- can capture nested structure

Cons:
- slower for large datasets
- sensitive to noise/outliers (depends on linkage)

---

### 3) DBSCAN (Density-Based Spatial Clustering for applications with noise)

DBSCAN forms clusters based on **density**.

Key parameters:
- `eps` ($\varepsilon$): neighborhood radius
- `min_samples`: minimum points needed to form a dense region

---

### Important Definitions

#### $\varepsilon$-Neighborhood

For a point $p$, its $\varepsilon$-neighborhood is:

$$
N_\varepsilon(p)=\{q\;|\;dist(p,q)\le \varepsilon\}
$$

---

#### Density-Reachable

A point $q$ is **density-reachable** from $p$ if:
- $q$ is within $\varepsilon$ of $p$
- and $p$ is a **core point**

More generally, $q$ is density-reachable from $p$ if there exists a chain:

$$
p=p_1, p_2, ..., p_k=q
$$

such that each $p_{i+1}$ is within $\varepsilon$ of $p_i$ and each $p_i$ (except maybe the last) is a **core point**.

Simple meaning:
- from a core point, you can “walk” to other points through dense regions
- relation is not symmetric

---

#### Density-Connected

Two points $p$ and $q$ are **density-connected** if there exists a point $o$ such that:
- $p$ is density-reachable from $o$
- $q$ is density-reachable from $o$

Simple meaning:
- both points belong to the same dense area (same cluster)
- relation is symmetric

---

#### Types of Points

- Core point: has at least `min_samples` within `eps`
- Border point: near a core point but not dense enough itself
- Noise / Outlier: not reachable from any core point

---

#### Steps (Intuition)

DBSCAN forms clusters by **expanding core points**.

1. Start with an unvisited point $p$
2. Find $N_\varepsilon(p)$ (neighbors inside radius $\varepsilon$)
3. If $p$ is not a core point (neighbors < `min_samples`)
- mark it as **noise** for now (it may later become a border point)
4. If $p$ is a core point
- create a new cluster
- add $p$ and all its neighbors to the cluster
5. Expand the cluster:
- for each neighbor point $q$ in the cluster:
  - find $N_\varepsilon(q)$
  - if $q$ is a core point, add its neighbors to the cluster too

Result:
- cluster = all points that are **density-reachable** from any core point in that cluster
- border points join the cluster if they are within $\varepsilon$ of a core point
- remaining points stay as noise/outliers

---

#### Pros / Cons

Pros:
- no need to choose `K`
- finds non-spherical clusters
- detects outliers naturally

Cons:
- hard to choose good `eps`
- struggles when cluster densities vary
- not great in very high dimensions (distance becomes less meaningful)

---

### 4) Gaussian Mixture Model (GMM)

GMM assumes data is generated from a mixture of **Gaussian distributions**.

Difference from K-Means:
- K-Means does **hard assignment** (point belongs to one cluster)
- GMM does **soft assignment** (probability of belonging to each cluster)

Example:

```text
Point x:
P(cluster1)=0.2
P(cluster2)=0.8
```

GMM is usually trained using EM (Expectation-Maximization).

---

### PCA (Principal Component Analysis)

PCA is an **unsupervised dimensionality reduction** technique.

Goal:
- convert many features into fewer new features (components)
- while keeping maximum information (variance)

PCA is commonly used before clustering to:
- reduce noise
- make distance-based methods (K-Means, DBSCAN) work better in high dimensions
- help visualization in 2D/3D

---

### Intuition

PCA finds new axes (directions) such that:
- **1st principal component (PC1)** captures maximum variance
- **2nd principal component (PC2)** captures next maximum variance
- PCs are **orthogonal** (90° to each other)

So PCA rotates the coordinate system to a better view of the data.

---

### Important Point (Scaling)

PCA is variance-based, so feature scaling is important.

Example:

```text
Age range: 0–60
Salary range: 0–500000
```

Salary will dominate variance if we do not scale.

So we usually do:
- Standardization (mean 0, std 1)

---

### Math Idea (High Level)

1. Center the data:

$$
X_{centered} = X - \mu
$$

2. Compute covariance matrix:

$$
\Sigma = \frac{1}{m-1} X_{centered}^T X_{centered}
$$

3. Find eigenvectors and eigenvalues:

$$
\Sigma v = \lambda v
$$

Where:
- eigenvectors $v$ = principal directions (PCs)
- eigenvalues $\lambda$ = variance captured by each PC

4. Choose top `k` eigenvectors (highest eigenvalues)

5. Project data onto these components:

$$
Z = X_{centered} W_k
$$

where $W_k$ contains the top `k` eigenvectors.

---

### Explained Variance

Explained variance ratio for component $j$:

$$
EVR_j = \frac{\lambda_j}{\sum_i \lambda_i}
$$

To choose `k`, we often pick enough components to cover:

```text
90% or 95% variance
```

---

### PCA Steps (Simple)

#### Step 0: Decide the goal

- For visualization: `k = 2` or `k = 3`
- For dimensionality reduction: choose `k` using explained variance (90% / 95%)

---

#### Step 1: Standardize / Scale features

For each feature $x$:

$$
x_{scaled} = \frac{x-\mu}{\sigma}
$$

Why:
- PCA depends on variance
- without scaling, large-scale features dominate

Output:
- scaled data matrix $X$ of shape $m \times n$ (m samples, n features)

---

#### Step 2: Mean-center the data

Compute mean of each feature and subtract it:

$$
X_{centered} = X - \mu
$$

Output:
- centered data (mean of each column becomes ~0)

---

#### Step 3: Compute covariance matrix

Covariance matrix tells how features vary together:

$$
\Sigma = \frac{1}{m-1}X_{centered}^T X_{centered}
$$

Shape:
- $\Sigma$ is $n \times n$

---

#### Step 4: Eigen decomposition (find PCs)

Solve:

$$
\Sigma v = \lambda v
$$

Where:
- $v$ (eigenvector) = direction of a principal component
- $\lambda$ (eigenvalue) = variance captured in that direction

Sort eigenvalues in decreasing order:

$$
\lambda_1 \ge \lambda_2 \ge ... \ge \lambda_n
$$

---

#### Step 5: Choose number of components `k` i.e no. of features in the transformed data

Compute explained variance ratio:

$$
EVR_j = \frac{\lambda_j}{\sum_i \lambda_i}
$$

Choose the smallest `k` such that:

$$
\sum_{j=1}^{k} EVR_j \ge 0.90 \;\; (or\;0.95)
$$

---

#### Step 6: Form projection matrix $W_k$

Take the top `k` eigenvectors and stack them as columns:

$$
W_k = [v_1\; v_2\; ...\; v_k]
$$

Shape:
- $W_k$ is $n \times k$

---

#### Step 7: Transform (project) the data

Project the centered data to `k` dimensions:

$$
Z = X_{centered}W_k
$$

Output:
- new dataset $Z$ of shape $m \times k$
- each column of $Z$ is a principal component

---

### Pros / Cons

Pros:
- reduces dimensions and noise
- speeds up training
- helps visualization
- reduces multicollinearity

Cons:
- loses interpretability (PCs are combinations of features)
- can lose information if `k` is too small
- PCA is linear (may not capture complex non-linear structure)


### Quick Comparison

```text
K-Means      → centroid-based, needs K
Hierarchical → tree-based, dendrogram cut
DBSCAN       → density-based, finds outliers, no K
GMM          → probabilistic (soft clusters), needs K
```

