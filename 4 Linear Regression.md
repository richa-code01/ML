### Regression

Technique used to find relationships between variables and predict outputs.

---

### Linear Regression

Predicts **continuous values** using a **best-fit straight line**.

#### Formula

$$
y = mx + b
$$

- `m` → slope (it measures for a unit change in x axis, how much change is done in y axis)
- `b` → intercept (from where does the line starts in y axis)

---

#### Hypothesis Function

$$
h_\theta(x) = \theta_0 + \theta_1x
$$

- `θ₀` → intercept (bias)
- `θ₁` → slope (weight)

![Hypothesis Function](https://editor.analyticsvidhya.com/uploads/27669loss%20function5.jpg)

```text
Experience → Model → Salary
```

---

#### Residual Error

Difference between:
- actual value
- predicted value

Best-fit line → minimum residual error.(value of weights and bias such that the error is minimum)

```text
Actual Point *
             |
             | Residual Error
             |
Predicted ---*----------------
          Best Fit Line
```

---

#### Cost Function

Measures total model error.

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2
$$

- `m` → training examples
- `hθ(xi)` → predicted value
- `yi` → actual value

- Squaring makes error positive
- Goal → minimize cost function

---

### Gradient Descent

Finds optimal:
- `θ₀`
- `θ₁`

#### Update Rule

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)
$$

- `α` → learning rate
- Small `α` → slow but stable
- Large `α` → may overshoot minima

#### Cost Curve

![Gradient Descent Curve](https://sassafras13.github.io/images/2019-12-11-LinReg-fig1.png)

- Lowest point → **Global Minima**

---

### Multiple Linear Regression

Used when multiple features exist.

#### Formula

$$
h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n
$$

```text
Area + Bedrooms + Location → House Price
```

---

### Applications

- Salary Prediction
- House Price Prediction
- Sales Forecasting

---

### Performance Metrics

Evaluate regression model performance.

---

### R² Score

Measures variance explained by model.

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

- `yi` → actual value
- `ŷi` → predicted value
- `ȳ` → mean value

- `R² = 1` → perfect fit
- `R² = 0.86` → explains 86% variance
- Higher `R²` → better model

---

### Problem with R²

Adding useless features can still increase `R²`.

```text
Location + Experience + Sleep Hours → Salary
```

- `Location` ✔ correlated
- `Experience` ✔ correlated
- `Sleep Hours` ✘ not correlated

Still, `R²` may increase.

---

### Adjusted R²

Penalizes unnecessary features.

$$
Adjusted\ R^2 = 1 - \left( \frac{(1-R^2)(n-1)}{n-p-1} \right)
$$

- `n` → rows/data points
- `p` → number of features

- Increases only for useful features
- Better than normal `R²`

---

### R² vs Adjusted R²

| Metric | Behavior |
|---|---|
| R² | Always increases with more features |
| Adjusted R² | Increases only for useful features |

---

### Overfitting

- Good on training data
- Bad on testing data

- Low Bias
- High Variance

```text
Overfitting

Data:  *  *   * *  *
Curve: ~~~~~ highly complex ~~~~~
```

Model memorizes training data.

---

### Underfitting

- Bad on training + testing data

- High Bias
- High Variance → Low Learning

```text
Underfitting

Data: *  *   * *  *
Line: -------------------
```

Model too simple to learn patterns.

---
### Best Case: low bias and low variance
  -Bias is simplicity of model. High bias means model is too simple not learning even basic patterns in data.
  -Variance is sensitivity to training data. High bias means model is memorizing training data, fails on new data.
  -Linear Regression has higher bias as we try to find a best fit straight line for simple linear regression and hyper plane(i.e plane surface)     for multiple linear regression.

### Ridge Regression (L2)

Adds penalty term to reduce overfitting.

#### Cost Function

$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2+\lambda\sum_{j=1}^{n}\theta_j^2
$$

- `λ` → regularization parameter
- `θj` → slope/weight

#### Effect

- Shrinks coefficients
- Reduces overfitting
- Smoothens best-fit line

---

### Lasso Regression (L1)

Uses absolute weights.

#### Cost Function

$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2+\lambda\sum_{j=1}^{n}|\theta_j|
$$

#### Effect

- Makes some coefficients `0`
- Performs feature selection

---

### Ridge vs Lasso

| Feature | Ridge | Lasso |
|---|---|---|
| Regularization | L2 | L1 |
| Formula | `θ²` | `θ` |
| Feature Selection | No | Yes |
| Overfitting Reduction | Yes | Yes |

---

### Logistic Regression

Classification algorithm for categorical outputs.

Creates a **decision boundary** between classes.

#### Sigmoid Function

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

#### Output Range

$$
0 \leq P(y) \leq 1
$$

#### Graph

```text
Probability
↑
1.0 |              ______
    |           __/
0.5 |__________/
    |
0.0 |___________________→ x
```

---

### Applications

- Spam Detection
- Disease Prediction
- Pass/Fail Prediction

---

### Linear vs Logistic Regression

| Feature | Linear | Logistic |
|---|---|---|
| Output | Continuous | Category |
| Graph | Straight Line | Sigmoid Curve |
| Used For | Regression | Classification |
