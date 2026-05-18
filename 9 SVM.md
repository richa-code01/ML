### 5. SVM (Support Vector Machine)

A supervised learning algorithm mainly used for:
- Classification
- Regression (SVR)

Goal:
- find the best separating boundary between classes
- maximize margin for better generalization

---

### Intuition

Suppose we have:
- Red balls
- Green balls

SVM tries to create a separating line/plane between them.

![SVM Hyperplane](https://media.geeksforgeeks.org/wp-content/uploads/20250807122724737293/support_vectors_hyperplane.webp)

---

### Important Terms

#### Hyperline

A separating line in **2D space**.

Example:
```text
x-axis + y-axis → 2D
```

---

#### Hyperplane

A separating boundary in **higher dimensions**.

Examples:
- 2D → line
- 3D → plane
- nD → hyperplane

Used to separate classes.

---

#### Margin

Distance between:
- hyperplane
- nearest data points

```text
Larger Margin = Better Generalization
```

---

#### Support Vectors

Closest points to the hyperplane.

These points:
- decide boundary position
- are most important data points

Removing them changes hyperplane.

---

### Visualization

```text
Red Class        Margin        Green Class
   ●  ●             |             ○   ○  ○
 ●   ●   ●          |              ○ ○ ○
   ●  ●             |            ○ ○  ○
---------------- Hyperplane ----------------
         ^                       ^
   Support Vector           Support Vector
```

---

### Hyperplane Equation

$$
w^Tx+b=0
$$

Where:
- `x` → input point/features
- `w` → weights/slopes
- `b` → bias/intercept

#### Bias Meaning

Bias shifts the hyperplane:
- left/right
- up/down

similar to intercept in linear regression.

---

### Prediction Rule

$$
\hat y = sign(w^Tx+b)
$$

#### Interpretation

If:

$$
w^Tx+b>0
$$

→ belongs to positive/red class

If:

$$
w^Tx+b<0
$$

→ belongs to negative/green class

---

### Hinge Loss Function

SVM commonly uses **Hinge Loss**.

#### Formula

$$
L=max(0,1-y(w^Tx+b))
$$

---

### Interpretation

- Correctly classified + outside margin:
  
```text
Loss = 0
```

- Misclassified / inside margin:

```text
Loss > 0
```

Goal:
- minimize hinge loss
- maximize margin

---

### Linear vs Non-Linear SVM

#### Linear SVM

Used when data is:
- linearly separable

Example:
```text
Straight line can separate classes
```

---

#### Non-Linear SVM

Used when:
- straight line cannot separate data

Solution:
```text
Kernel Trick
```

---

### Kernel Trick

Maps data into higher dimensions
so separation becomes easier.

#### Data Dimension

Dimension = number of input features.

Example:

```text
Height + Weight
```

→ 2D data

```text
Height + Weight + Age
```

→ 3D data

Higher dimensions can help separation.

---

### Common Kernels

#### 1. Linear Kernel

Used for:
- linearly separable data
- text classification

Fastest kernel.

---

#### 2. Polynomial Kernel

Creates curved boundaries.

Useful when relationships are polynomial.

---

### 3. RBF / Gaussian Kernel

Most commonly used kernel.

Creates highly flexible non-linear boundaries.

Works well for complex datasets.

#### Formula

$$
K(x_i,x_j)=\exp(-\gamma ||x_i-x_j||^2)
$$

Where:
- `xᵢ , xⱼ` → two data points
- `||xᵢ - xⱼ||²` → squared Euclidean distance
- `γ` (gamma) → controls influence of points
- `exp` → exponential function

---

### Intuition

- Nearby points → high similarity
- Far points → low similarity

RBF maps data into higher dimensions
where linear separation becomes possible.

---

### Gamma (γ) in RBF Kernel

Controls influence region of each point.

#### Small γ

- smoother boundary
- wider influence
- less overfitting

```text
Large smooth curves
```

#### Large γ

- very complex boundary
- narrow influence
- may overfit

```text
Highly wiggly boundary
```

### Hyperparameter C

Controls tradeoff between:
- margin size
- classification error

#### Small C

- larger margin
- allows some misclassification
- less overfitting

#### Large C

- smaller margin
- strict classification
- may overfit

---

### Steps to Build SVM

1. Prepare dataset `(X,y)`
2. Perform feature scaling
3. Choose kernel:
   - Linear
   - Polynomial
   - RBF
4. Tune:
   - `C`
   - `gamma`
   - `degree`
5. Train + evaluate model

Metrics:
- Accuracy
- Precision
- Recall
- F1 Score

---

### Feature Scaling

Very important in SVM because:
- SVM depends on distances/margins

Without scaling:
- larger features dominate calculations

Example:

```text
Age = 20
Salary = 200000
```

Salary dominates distance.

---

### Multiclass SVM

SVM is naturally binary.

For multiclass:
- One-vs-Rest (OvR)
- One-vs-One (OvO)

---

### Advantages

- Works well in high-dimensional data
- Effective for complex boundaries
- Good generalization
- Memory efficient

---

### Disadvantages

- Slow on very large datasets
- Sensitive to feature scaling
- Kernel selection can be difficult
- Less interpretable

---

### Applications

- Spam Detection
- Image Classification
- Handwriting Recognition
- Face Detection
- Bioinformatics
- Sentiment Analysis

---

### Important Interview Points

- SVM maximizes margin
- Support vectors define hyperplane
- Feature scaling is important
- RBF kernel is most commonly used
- `gamma` controls boundary smoothness
- `C` controls margin vs misclassification tradeoff
