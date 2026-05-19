### 11. Ensemble Learning

Ensemble learning means combining multiple models to make a **better final prediction** than a single model.

In simple words:
- one model may make mistakes
- many models together can reduce mistakes

---

### Why Ensemble Works?

If individual models make different errors, combining them can:
- reduce **variance** (more stable)
- sometimes reduce **bias** (more accurate)

Goal:
- improve generalization on unseen data

---

### How Predictions Are Combined

#### 1) Voting (Classification)

##### Hard Voting
- final class = majority vote

Example:

```text
Model 1: Yes
Model 2: No
Model 3: Yes

Final = Yes (2/3)
```

##### Soft Voting
- take average of predicted probabilities
- choose class with highest probability

Example:

```text
P(Yes) = (0.7 + 0.4 + 0.8) / 3 = 0.63
P(No)  = 1 - 0.63 = 0.37

Final = Yes
```

---

#### 2) Averaging (Regression)

Final prediction = average of predictions:

$$
\hat{y}=\frac{\hat{y}_1+\hat{y}_2+...+\hat{y}_n}{n}
$$

---

### Types of Ensemble Methods

#### 1) Bagging (Bootstrap Aggregation)

Idea:
- train many instances of the same model/algorithm on different random samples of data
- combine their predictions (vote/average)

Sampling method:
- **bootstrap** sampling = sample with replacement

Why bagging is used:
- mainly reduces **variance**
- helps unstable models (like decision trees)

---

### Random Forest (Bagging + Feature Randomness)

Random Forest is an ensemble of many **decision trees**.

Key points:
- each tree is trained on a bootstrap sample
- at each split, tree considers only a random subset of features
- post that we do majority voting / mean of the output of these decision trees to predict the final output.

Why it performs well:
- trees become less correlated
- majority voting reduces overfitting

Pros:
- good accuracy
- handles non-linear data
- less overfitting than a single tree

Cons:
- less interpretable
- can be slower and larger

---

#### 2) Boosting

Boosting means:
- build many weak models **one after another (sequentially)**
- each new model tries to fix the mistakes made by previous models

In simple words:
- Model 1 makes some errors
- Model 2 learns mainly from those errors
- Model 3 fixes what is still wrong
- final prediction = combination of all models

Why boosting is used:
- reduces **bias** and improves accuracy
- can also reduce error step-by-step

---

### AdaBoost (Adaptive Boosting)

Idea (easy):
- give more importance (weight) to the points that were predicted wrong
- next weak model focuses more on those difficult points

How it “focuses on error points”:
1. Start: all data points have equal weight
2. Train a weak learner (usually a small decision tree stump)
3. Increase weights of misclassified points
4. Train next learner on the re-weighted data

Final prediction:
- weighted vote of weak learners

---

### Gradient Boosting (GB)

Idea (easy):
- think of it like correcting your answer step-by-step
- each new model learns the **remaining error** (called residual)

How it “corrects past errors”:
1. Start with a simple prediction

```text
Prediction 1: everyone gets average value
```

2. Compute errors (residuals)

```text
Residual = Actual - Predicted
```

3. Train next model to predict these residuals
4. Update prediction:

$$
New\ Prediction = Old\ Prediction + (learning\ rate)\times (model\ predicting\ residual)
$$

So every next model improves the previous prediction.

---

### XGBoost (Extreme Gradient Boosting)

XGBoost is a fast and powerful implementation of **Gradient Boosting using decision trees**.

Why it is popular:
- adds **regularization** (helps reduce overfitting)
- uses **learning rate (shrinkage)** to take small safe steps
- supports **row/column sampling** (faster + less overfitting)
- handles **missing values** smartly
- optimized for speed (parallelization + efficient code)

In simple words:
- it is still gradient boosting
- but engineered to be faster, more accurate, and more stable

---

#### 3) Stacking (Stacked Generalization)

Idea:
- train multiple different **base models**
- use their predictions as input features for a **meta-model**
- meta-model learns how to combine base model outputs to give final output

In simple words:
- base models are like different “opinions”
- meta-model learns which opinion to trust more i.e. which model's output is given more weights.

---

### How Stacking Combines (Easy)

Stacking happens in 2 phases.

#### Phase 1: Create “new dataset” for meta-model (Training)

For each training sample, collect predictions from base models.

Example (churn, prediction = `P(Yes)`):

```text
Sample   LR     SVM    Tree    True(Y)
S1      0.72   0.55   0.90       1
S2      0.20   0.30   0.10       0
S3      0.60   0.65   0.40       1
```

Now meta-model gets inputs like:

```text
X_meta = [LR_pred, SVM_pred, Tree_pred]
Y_meta = True label
```

Important:
- while creating these predictions for training, we usually use CV (OOF predictions)
- so a base model does not predict on a sample it already trained on

---

#### Phase 2: Final Prediction (Testing / New Data)

For a new sample `x`:
1. Base models predict:

```text
LR(x)=0.72,  SVM(x)=0.55,  Tree(x)=0.90
```

2. Meta-model takes these as input and combines them.

Simple combination example (meta-model learns weights):

$$
P(Yes)=0.5\times 0.72 + 0.3\times 0.55 + 0.2\times 0.90
$$

If $P(Yes) > 0.5$ then final class = `Yes`.

---

### Bagging vs Boosting (Quick Difference)

Bagging:
- models are trained in parallel
- mainly reduces variance
- example: Random Forest

Boosting:
- models are trained one after another
- reduces bias and improves accuracy
- examples: AdaBoost, Gradient Boosting

---

### When to Use Ensemble?

Use ensemble when:
- single model is unstable or overfitting
- you need better accuracy

Avoid/limit ensemble when:
- you need very high interpretability
- model size/speed is a strict constraint

