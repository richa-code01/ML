### What is Classification?

Used to predict a **category/class label** for given input data.

#### Examples

- Cat or Dog?
- Spam or Not Spam?
- Customer will Leave or Stay?
- Obesity Exists or Not?

Output can be:
- Yes / No
- True / False
- Multiple categories

---

### Logistic Regression

Used for **classification problems** (mostly binary classification).

Predicts probabilities using an **S-shaped sigmoid curve**.

#### Linear vs Logistic Regression

- Linear Regression → straight line
- Logistic Regression → sigmoid (S-shaped) curve

![Linear vs Logistic](https://miro.medium.com/0*i5TzGZ03vn7hNMvj.png)

---

### Why not Straight Line?

Linear regression is not suitable for classification because:

#### Problems

1. Outliers influence the straight line heavily  
2. Predictions can become:
   - `< 0`
   - `> 1`

But probabilities should always lie in:

$$
[0,1]
$$

---

### Solution → Sigmoid Function

We squash the straight line into an S-shaped curve using **Sigmoid Activation Function**.

#### Sigmoid Function

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

- Output range:

$$
0 \leq \sigma(x) \leq 1
$$

- `e` → Euler's constant

$$
e = 2.71828
$$

`e` helps create smooth curves while squashing the line.

---

### Hypothesis Function

#### Linear Regression

$$
h_\theta(x)=\theta_0+\theta_1x_1
$$

#### Logistic Regression

$$
h_\theta(x)=g(\theta_0+\theta_1x_1)
$$

where:

$$
g(x)=\frac{1}{1+e^{-x}}
$$

Final equation:

$$
h_\theta(x)=\frac{1}{1+e^{-(\theta_0+\theta_1x_1)}}
$$

---

### Decision Boundary

- If probability `> 0.5` → Class = `1`
- If probability `< 0.5` → Class = `0`

#### Example

```text
P(Obesity) > 50%  → Obesity Exists
P(Obesity) < 50%  → No Obesity
```

---

### Sigmoid Curve

![Sigmoid Function](https://media.geeksforgeeks.org/wp-content/uploads/20250131185746649092/Sigmoid-Activation-Function.png)
---

### Cost Function (Log Loss / Binary Cross Entropy)

Measures how close predicted probabilities are to actual labels.

---

#### For Single Observation

$$
Loss = -(y\log(\hat y)+(1-y)\log(1-\hat y))
$$

---

#### For Entire Dataset

$$
LogLoss=-\frac{1}{N}\sum_{i=1}^{N}[y_i\log(\hat y_i)+(1-y_i)\log(1-\hat y_i)]
$$

---

### Notations

- `N` → number of samples
- `yᵢ` → actual label (`0` or `1`)
- `ŷᵢ` → predicted probability

---

### Interpretation

#### Case 1 → Correct Prediction

```text
Actual = 1
Predicted = 0.99
```

Very small loss.

---

#### Case 2 → Wrong Prediction

```text
Actual = 1
Predicted = 0.01
```

Very high loss.

---

### Why Log Loss Works?

Log loss gives:
- small penalty for correct confident predictions
- huge penalty for wrong confident predictions

#### Example

```text
Predicted = 0.99 , Actual = 1
Loss ≈ very low
```

```text
Predicted = 0.01 , Actual = 1
Loss ≈ very high
```

So the model learns to predict probabilities correctly.

---

### Applications

- Spam Detection
- Disease Prediction
- Fraud Detection
- Customer Churn Prediction
- Obesity Prediction
