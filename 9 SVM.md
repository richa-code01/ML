
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
$$

### Interpretation

- If \( w^T x + b > 0 \)

  → the point lies on the **right/top side** of the hyperplane  
  → the positive class is predicted

- If \( w^T x + b < 0 \)

  → the point lies on the **left/below side** of the hyperplane  
  → the negative class is predicted

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

Common kernels:

#### 1) Linear Kernel

#### 2) Polynomial Kernel

#### 3) RBF / Gaussian Kernel (Most common)

### Role of γ (Gamma) in RBF

---

### Steps to Build an SVM Model

1. Prepare dataset (X, y)
2. **Feature scaling is important as we calculate margin** (especially for RBF kernel)
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
- `gamma` controls smoothness in RBF kernel
- Always scale features for distance-based kernels

