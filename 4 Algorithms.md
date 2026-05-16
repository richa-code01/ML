# Regression

Technique used to find relationships between variables and predict outputs.

---

# Linear Regression

Predicts **continuous values** using a **best-fit straight line**.

## Formula

$$
y = mx + b
$$

- `m` → slope
- `b` → intercept

---

# Hypothesis Function

$$
h_\theta(x) = \theta_0 + \theta_1x
$$

- `θ₀` → intercept (bias)
- `θ₁` → slope (weight)
- 
![Hypothesis Function](https://editor.analyticsvidhya.com/uploads/27669loss%20function5.jpg)

## Example

```text
Experience → Model → Salary
```

---

# Residual Error

Difference between:
- actual value
- predicted value

Best-fit line → minimum residual error.

```text
Actual Point *
             |
             |  Residual Error
             |
Predicted ---*----------------
          Best Fit Line
```

---

# Cost Function

Measures total model error.

## Formula

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2
$$

- `m` → number of training examples
- `hθ(xi)` → predicted value
- `yi` → actual value

## Notes

- Squaring makes error positive
- Goal → minimize cost function

---

# Gradient Descent

Used to find values of:
- `θ₀`
- `θ₁`

that minimize cost function.

## Update Rule

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)
$$

- `α` → learning rate

## Learning Rate

- Small `α` → slow but stable learning
- Large `α` → may overshoot minima

## Cost Curve

![Gradient Descent Curve](https://sassafras13.github.io/images/2019-12-11-LinReg-fig1.png)

- Lowest point → **Global Minima**
- Gradient Descent moves towards minimum cost.

---

# Multiple Linear Regression

Used when multiple input features exist.

## Formula

$$
h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n
$$

## Example

```text
Area + Bedrooms + Location → House Price
```

---

# Applications

- Salary Prediction
- House Price Prediction
- Sales Forecasting

---

# Performance Metrics

Used to evaluate how well a regression model performs.

---

# R² Score (Coefficient of Determination)

Measures how well the model explains the variance in data.

## Formula

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

Where:
- `yi` → actual value
- `ŷi` → predicted value
- `ȳ` → mean of actual values

---

## Interpretation

- `R² = 1` → perfect prediction
- `R² = 0` → model explains nothing
- `R² = 0.86` → model explains 86% variance

## Notes

- Value lies between `0` and `1`
- Higher `R²` → better model

---

# Problem with R²

Adding more features can increase `R²`
even if those features are irrelevant.

## Example

```text
Location + Experience + Sleep Hours → Salary
```

- `Location` ✔ correlated
- `Experience` ✔ correlated
- `Sleep Hours` ✘ not correlated

Still, `R²` may increase.

---

# Adjusted R²

Penalizes unnecessary features.

Used to check whether added features are actually useful.

## Formula

$$
Adjusted\ R^2 = 1 - \left( \frac{(1-R^2)(n-1)}{n-p-1} \right)
$$

Where:
- `n` → number of rows/data points
- `p` → number of features

---

## Notes

- Increases only if new feature improves model significantly
- Prevents overfitting
- More reliable than normal `R²`

---

# R² vs Adjusted R²

| Metric | Behavior |
|---|---|
| R² | Always increases with more features |
| Adjusted R² | Increases only for useful features |

# Logistic Regression

Classification algorithm used for categorical outputs.

Creates a **decision boundary** between classes.

## Sigmoid Function

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

## Output Range

$$
0 \leq P(y) \leq 1
$$

## Graph

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

# Applications

- Spam Detection
- Disease Prediction
- Pass/Fail Prediction

---

# Linear vs Logistic Regression

| Feature | Linear | Logistic |
|---|---|---|
| Output | Continuous | Category |
| Graph | Straight Line | Sigmoid Curve |
| Used For | Regression | Classification |
