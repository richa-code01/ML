### Ensemble Learning

Ensemble Learning means:
```text
Combining multiple models to get better prediction
```

Idea:
- one model may make mistakes
- many models together reduce mistakes

Like:
```text
Listening to crowd opinion
```

---

### Why Ensemble Works?

Combining models can:
- reduce variance → less overfitting
- reduce bias → more accurate
- improve generalization on unseen data

---

### Basic Flow

```text
Train/Test Data
        ↓
 Multiple Models
        ↓
 Combine Predictions
        ↓
 Final Prediction
```

---

### Combining Predictions

#### Classification → Voting

##### Hard Voting

Majority class wins.

Example:

```text
Model 1 → Yes
Model 2 → No
Model 3 → Yes

Final → Yes
```

---

##### Soft Voting

Average probabilities.

Example:

```text
P(Yes)=(0.7+0.4+0.8)/3=0.63
```

```text
Final → Yes
```

---

#### Regression → Averaging

Take mean of predictions.

$$
\hat y=\frac{\hat y_1+\hat y_2+...+\hat y_n}{n}
$$

---

### Types of Ensemble Learning

```text
1. Bagging
2. Boosting
3. Stacking
```

---

## 1. Bagging (Bootstrap Aggregation)

Idea:
- train same algorithm multiple times
- use different random samples
- combine outputs

Main Goal:
```text
Reduce Variance / Overfitting
```

---

### Bootstrap Sampling

Sampling:
```text
with replacement
```

Meaning:
- same datapoint can repeat
- some datapoints may be absent

---

### Example

Dataset:
```text
2000 datapoints
```

Create bootstrap samples:

```text
D1 = 1000 random samples
D2 = 1000 random samples
D3 = 1000 random samples
D4 = 1000 random samples
```

Data can overlap,
but samples are not identical.

---

### Training Example

```text
       Different Random Samples
      /      |      |      \
     D1      D2     D3     D4
      ↓       ↓      ↓      ↓
     M1      M2     M3     M4
    SVM     SVM    SVM    SVM
```

---

### Final Prediction

#### Classification

```text
Majority Voting
```

Example:

```text
M1 → Yes
M2 → No
M3 → Yes
M4 → Yes

Final → Yes
```

---

#### Regression

```text
Take Mean Prediction
```

---

### Random Forest

Most popular bagging algorithm.

Random Forest:
```text
Collection of Decision Trees
```

---

### Random Forest Working

Each tree:
- trained on bootstrap sample
- uses random subset of features

Final prediction:
- majority voting
- or averaging

---

### Why Random Forest Works?

Trees become:
```text
Less Correlated
```

Thus:
```text
Less Overfitting
```

---

### Advantages

- good accuracy
- handles non-linear data
- robust to overfitting

---

### Disadvantages

- less interpretable
- larger/slower model

---

## 2. Boosting

Idea:
```text
Train models sequentially
```

Each next model:
```text
Learns from previous mistakes
```

---

### Flow

```text
M1 → M2 → M3
```

---

### Intuition

```text
M1 makes errors
↓
M2 focuses on those errors
↓
M3 fixes remaining mistakes
```

---

### Goal

- reduce bias
- improve accuracy
- avoid underfitting

---

## AdaBoost (Adaptive Boosting)

Idea:
```text
Hard/Error points get higher importance
```

---

### Working

#### Step 1

Initially:
```text
All datapoints have equal weights
```

---

#### Step 2

Train weak learner.

Usually:
```text
Decision Stump
```

(small decision tree)

---

#### Step 3

Wrong predictions:
```text
Weight Increased
```

Correct predictions:
```text
Weight Decreased
```

---

#### Step 4

Next model focuses more on:
```text
Hard datapoints
```

---

### Final Prediction

```text
Weighted Voting
```

---

## Gradient Boosting

Idea:
```text
Every new model learns residual errors
```

Residual:

$$
Residual=Actual-Predicted
$$

---

### Working

#### Step 1

Initial prediction:

```text
Predict average value
```

---

#### Step 2

Find errors:

```text
Residual = Actual - Predicted
```

---

#### Step 3

Next model learns:
```text
Residuals
```

---

#### Step 4

Update prediction:

$$
New\ Prediction=
Old\ Prediction+
LearningRate\times ResidualModel
$$

---

### Result

Every next model:
```text
Improves previous prediction
```

---

## XGBoost (Extreme Gradient Boosting)

Advanced implementation of:
```text
Gradient Boosting
```

Uses:
- decision trees
- regularization
- optimized training

---

### Why XGBoost Popular?

- very fast
- highly accurate
- handles overfitting
- supports parallel processing
- handles missing values

---

### Important Features

#### Regularization

Helps:
```text
Reduce Overfitting
```

---

#### Learning Rate (Shrinkage)

Takes:
```text
Small Safe Steps
```

---

#### Row/Column Sampling

- faster
- less overfitting

---

## 3. Stacking (Stacked Generalization)

Idea:
- train multiple different models
- use their predictions as input to another model

Final model:
```text
Meta Model
```

learns:
```text
Which model to trust more
```

---

### Stacking Flow

```text
                Data
                  ↓
      ┌────────┬────────┬────────┐
      ↓        ↓        ↓
     SVM      DT      LogReg
      ↓        ↓        ↓
      1        0        1
            Predictions
                 ↓
          Meta Model (KNN)
                 ↓
            Final Output
```

---

### Intuition

Base models:
```text
Different Opinions
```

Meta-model:
```text
Learns best combination
```

---

## Phase 1 → Meta Training Dataset

For every training sample:
- collect predictions from base models

Example:

```text
Sample   LR    SVM   Tree   True(Y)
S1      0.72   0.55   0.90     1
S2      0.20   0.30   0.10     0
S3      0.60   0.65   0.40     1
```

Meta dataset:

```text
X_meta=[LR,SVM,Tree predictions]
Y_meta=True Label
```

---

### Important

Usually:
```text
OOF Predictions (Out-of-Fold)
```

using:
```text
Cross Validation
```

so model does not predict on data it trained on.

---

## Phase 2 → Final Prediction

For new datapoint:

```text
LR(x)=0.72
SVM(x)=0.55
Tree(x)=0.90
```

Meta-model combines them.

Example:

$$
P(Yes)=0.5(0.72)+0.3(0.55)+0.2(0.90)
$$

If:

$$
P(Yes)>0.5
$$

Final:
```text
YES
```

---

### Bagging vs Boosting

| Feature | Bagging | Boosting |
|---|---|---|
| Training | Parallel | Sequential |
| Goal | Reduce Variance | Reduce Bias |
| Overfitting | Reduced | Can Overfit |
| Example | Random Forest | AdaBoost/XGBoost |

---

### When to Use Ensemble?

Use when:
- single model unstable
- overfitting occurs
- better accuracy needed

Avoid when:
- interpretability important
- strict latency/small model required

---

### Important Interview Points

- Ensemble = combine multiple models
- Bagging → parallel training
- Boosting → sequential learning
- Stacking → meta-model combines outputs
- Random Forest = bagging of trees
- XGBoost = optimized gradient boosting
- Bagging reduces variance
- Boosting reduces bias
