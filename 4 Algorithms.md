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
