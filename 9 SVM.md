
### 5. SVM (Support Vector Machine)

A supervised learning algorithm used mainly for:
- Classification
- (Also works for regression → SVR)

Goal: find a **decision boundary (hyperplane)** that separates classes with the **maximum margin**.

---

### Key Terms

- **Hyperplane/Hyperline**: the boundary/line that separates classes.
- **Margin**: distance between hyperplane and the closest points from each class.
- **Support Vectors**: the closest data points to the hyperplane.

```text
Wider Margin  => better generalization
Support Vectors => only points that matter most
```

---

### Intuition (2D)

For 2 features, the boundary is a line:

```text
Class +      |     Class -
   +  +      |   -   -
      +      |     -
-------------(hyperplane)-------------
       ^          ^
   support     support
   vector      vector
```

SVM tries to place the line so that the margin is maximum.

---

### Hyperplane Equation

In n-dimensions:

$$
w^T x + b = 0
$$

- $w$ → weights (normal vector)
- $b$ → bias

Prediction rule:

$$
\hat y = sign(w^T x + b)
if y is positive right/top of the hyperplane class is predicted else left/below class is predicted. 
$$

---

### Maximum Margin Objective (Hard Margin)

If data is perfectly separable:

Minimize:

$$
\frac{1}{2}||w||^2
$$

Subject to (for labels $y_i \in \{+1,-1\}$):

$$
y_i(w^T x_i + b) \ge 1
$$

Margin is:

$$
	ext{Margin} = \frac{2}{||w||}
$$

So minimizing $||w||$ maximizes margin.

---

### Soft Margin SVM (Handles Noise / Overlap)

Real-world data is not perfectly separable.

We allow some violations using slack variables $\xi_i$.

Minimize:

$$
\frac{1}{2}||w||^2 + C\sum_{i=1}^{m}\xi_i
$$

Subject to:

$$
y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0
$$

---

### Role of C (Regularization)

`C` controls the trade-off:

- Large `C` → tries to classify all training points correctly
  - low bias, high variance (can overfit)
- Small `C` → allows more misclassification
  - higher bias, lower variance (more generalization)

```text
C ↑  => narrower margin, fewer errors allowed
C ↓  => wider margin, more errors allowed
```

---

### Loss Function (Hinge Loss)

SVM is commonly explained using hinge loss.

For one sample:

$$
L = max(0, 1 - y(w^T x + b))
$$

- If correctly classified with margin ≥ 1 → loss = 0
- If inside margin / misclassified → positive loss

---

### Linear vs Non-Linear SVM

#### Linear SVM

Use when data is roughly linearly separable.

#### Non-Linear SVM

Use when classes need a curved boundary. i.e classes are not linearly separable via plane/line.

Solution → **Kernel Trick**.

---

### Kernel Trick

Idea: map input to higher dimension so separation becomes linear.

Instead of explicitly creating new features, SVM uses a kernel function:

$$
K(x, z) = \phi(x)^T\phi(z)
$$

Common kernels:

#### 1) Linear Kernel

$$
K(x,z) = x^T z
$$

#### 2) Polynomial Kernel

$$
K(x,z) = (\gamma x^T z + r)^d
$$

#### 3) RBF / Gaussian Kernel (Most common)

$$
K(x,z) = exp(-\gamma ||x-z||^2)
$$

---

### Role of γ (Gamma) in RBF

- Large $\gamma$ → each point has small influence region
  - complex boundary (overfitting risk)
- Small $\gamma$ → smoother boundary
  - underfitting risk

```text
gamma ↑  => more wiggly boundary
gamma ↓  => smoother boundary
```

---

### Steps to Build an SVM Model

1. Prepare dataset (X, y)
2. **Feature scaling is important** (especially for RBF kernel)
3. Choose kernel:
   - linear (fast, high-dimensional sparse)
   - RBF (general purpose)
4. Tune hyperparameters:
   - `C`
   - `gamma` (for RBF)
   - `degree` (for polynomial)
5. Train model and evaluate (Accuracy / Precision / Recall / F1)

---

### Multiclass SVM

SVM is naturally binary, but supports multiclass using:

- One-vs-Rest (OvR)
- One-vs-One (OvO)

---

### Advantages

- Works well in high-dimensional space
- Effective when number of features > number of samples
- Good margin-based generalization
- Uses only support vectors (memory efficient after training)

---

### Disadvantages

- Training can be slow on very large datasets
- Sensitive to feature scaling
- Kernel + hyperparameter selection can be tricky
- Less interpretable than linear/logistic regression

---

### Applications

- Text classification (spam, sentiment)
- Image classification (small/medium datasets)
- Bioinformatics (gene classification)
- Handwriting recognition

---

### Important Interview Points

- SVM maximizes margin; boundary depends on **support vectors**
- `C` controls margin vs training errors
- `gamma` controls smoothness in RBF kernel
- Always scale features for distance-based kernels

