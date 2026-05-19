### Model Tuning (Hyperparameter Tuning)

Improving model performance on:
- unseen data
- validation/test data

We:
- do NOT change dataset
- do NOT change algorithm
- only tune hyperparameters

Goal:

```text
Better Generalization
```

---

### Hyperparameters

Values set before training.

They control:
- learning behavior
- model complexity

| Model | Hyperparameters |
|---|---|
| KNN | `n_neighbors (k)` |
| Decision Tree | `max_depth`, `min_samples_split` |
| SVM | `C`, `kernel`, `gamma` |
| Random Forest | `n_estimators` |
| Ridge/Lasso | `lambda (α)` |

> Parameters are learned from data, hyperparameters are chosen manually.

---

### Basic Workflow

```text
Dataset
   ↓
Train Model
   ↓
Try Multiple Hyperparameters
   ↓
Evaluate
   ↓
Select Best Combination
```

---

### Data Split

```text
Training Set   → learning
Validation Set → tuning
Test Set       → final evaluation
```

> Never tune on test set → data leakage.

---

### Example Hyperparameters

#### KNN

```text
k = 1,3,5,7,9
```

#### SVM

```text
C = 0.1,1,10
kernel = linear,rbf
```

---

### Evaluation Metrics

Classification:
- Accuracy
- Precision
- Recall
- F1 Score

Regression:
- MAE
- MSE
- RMSE
- R²

---

### Cross Validation (CV)

Problem:

```text
What if current train-validation split is bad?
```

Maybe:
- overfitting
- underfitting
- biased score

Solution:
```text
K-Fold Cross Validation
```

---

### K-Fold Cross Validation

Split dataset into:
```text
K equal folds
```

Usually:

```text
K = 5
```

![K Fold CV](https://editor.analyticsvidhya.com/uploads/57458kzw7d4mp9b45VFQN2wjKu1K8J9KrDh.png)

---

### Working

```text
F1 F2 F3 F4 F5
```

#### Runs

```text
Run 1 → Train(F2,F3,F4,F5) Validate(F1)

Run 2 → Train(F1,F3,F4,F5) Validate(F2)

Run 3 → Train(F1,F2,F4,F5) Validate(F3)

Run 4 → Train(F1,F2,F3,F5) Validate(F4)

Run 5 → Train(F1,F2,F3,F4) Validate(F5)
```

---

### Final CV Score

Average of all validation scores.

Example:

```text
0.82,0.80,0.84,0.81,0.83
```

$$
CV\ Score=\frac{0.82+0.80+0.84+0.81+0.83}{5}
$$

---

### Why CV?

#### Advantages

- reliable evaluation
- every sample validated once
- better data usage

#### Disadvantages

- slower
- trains model K times

> Stratified K-Fold is preferred for imbalanced datasets.

---

### Hyperparameter Tuning Methods

---

#### 1. Manual Search

Try values manually.

Example:

```text
max_depth = 2,4,6,8
```

Workflow:
```text
Try → Train → Evaluate → Repeat
```

##### Pros

- simple
- fast for small tasks

##### Cons

- not systematic
- may miss best combination

---

#### 2. Grid Search CV

Tries:
```text
ALL possible combinations
```

with:
```text
Cross Validation
```

---

### Example Grid

```text
n_neighbors = [3,5,7]
weights = [uniform,distance]
metric = [euclidean,manhattan]
```

---

### What Happens?

```text
(3,uniform,euclidean)
(3,uniform,manhattan)
(3,distance,euclidean)
...
```

Each combination:
- trained
- validated using CV

Best average score:
```text
→ Best Hyperparameters
```

---

### Training Cost

$$
Total\ Trainings=(Combinations)\times K
$$

Can become slow for large grids.

---

##### Pros

- systematic
- strong baseline

##### Cons

- computationally expensive

---

#### 3. Randomized Search CV

Instead of checking all combinations:
```text
randomly samples combinations
```

Faster for large search spaces.

---

### Example

```text
C = [0.001,0.01,0.1,1,10]
gamma = [0.001,0.01,0.1,1]
```

Randomly tests:
```text
N combinations
```

Example:

```text
N = 20
```

---

### Working

1. define parameter ranges
2. randomly sample combinations
3. train + validate
4. choose best score

---

### Training Cost

$$
Total\ Trainings=N\times K
$$

---

##### Pros

- much faster
- works well for large search spaces

##### Cons

- may miss best combination

---

### Grid vs Random Search

![Grid vs Random Search](https://miro.medium.com/v2/resize:fit:1200/1*WOB7yoQMg6IE0CiO0_l6bg.png)

| Feature | Grid Search | Random Search |
|---|---|---|
| Checks All Combinations | Yes | No |
| Speed | Slow | Faster |
| Large Search Space | Poor | Good |

---

### Important Interview Points

- Hyperparameters are manually chosen
- Parameters are learned from data
- CV gives reliable evaluation
- Grid Search = exhaustive search
- Random Search = random sampling
- K-Fold retrains model K times
