### Misc Notes

---

### How a Model Actually Learns (Weights + Biases)

Most trainable models learn by:
- making predictions (forward pass)
- measuring error using a loss function
- adjusting parameters (weights and biases) to reduce the loss

---

### 1) A Simple Example (Linear Model)

For 1 input feature $x$:

$$
\hat{y} = wx + b
$$

Where:
- $w$ = weight (slope)
- $b$ = bias (intercept)

For many features $x\in\mathbb{R}^n$:

$$
\hat{y} = w^Tx + b
$$

---

### 2) Loss Function (How wrong the model is)

For regression, common loss is Mean Squared Error (MSE):

$$
L(w,b)=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

Goal:

$$
\min_{w,b} \; L(w,b)
$$

---

### 3) Gradient (Direction to Reduce Loss)

Gradient tells:
- how loss changes when we slightly change $w$ or $b$

Compute partial derivatives:

$$
\frac{\partial L}{\partial w},\;\;\frac{\partial L}{\partial b}
$$

For the 1D linear regression case ($\hat{y}=wx+b$):

$$
\frac{\partial L}{\partial w}=\frac{-2}{m}\sum_{i=1}^{m}x_i\,(y_i-\hat{y}_i)
$$

$$
\frac{\partial L}{\partial b}=\frac{-2}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)
$$

Intuition:
- if prediction is too low ($y-\hat{y}$ positive), update increases $w/b$
- if prediction is too high ($y-\hat{y}$ negative), update decreases $w/b$

---

### 4) Gradient Descent Update Rule

Update parameters in the opposite direction of gradient:

$$
w \leftarrow w - \alpha\frac{\partial L}{\partial w}
$$

$$
b \leftarrow b - \alpha\frac{\partial L}{\partial b}
$$

Where:
- $\alpha$ = learning rate (step size)

Repeat many times (epochs) until loss becomes small.

---

### 5) What is Backpropagation?

Backpropagation is an efficient way to compute gradients in a multi-layer neural network.

One layer (forward):

$$
z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = f(z^{(l)})
$$

Loss depends on many parameters:

$$
L = L(W^{(1)},b^{(1)},W^{(2)},b^{(2)},...)
$$

Backprop uses chain rule to send “error signal” from output to earlier layers:

$$
\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial a^{(L)}}\cdot \frac{\partial a^{(L)}}{\partial a^{(L-1)}}\cdot ... \cdot \frac{\partial a^{(2)}}{\partial W^{(1)}}
$$

So training looks like:
- forward pass → compute $\hat{y}$
- compute loss $L$
- backward pass (backprop) → compute gradients
- gradient descent → update weights/biases

---

### ML vs DL (Core Difference)

Both ML and DL minimize a loss, but the main difference is:
- **ML** often needs manual feature engineering
- **DL** learns features automatically from raw data

---

### ML vs DL (Difference based on Layers)

Important:
- the “number of layers” comparison mainly makes sense for **neural networks**
- many ML algorithms (trees, KNN) are not layer-based

---

### 1) If we talk only about Neural Networks

#### Shallow Model (often called ML)

- no hidden layer (linear/logistic regression style):

$$
\hat{y}=f(w^Tx+b)
$$

```text
Input  →  Output
```

- one hidden layer (still shallow):

```text
Input  →  Hidden  →  Output
```

---

#### Deep Learning (DL)

Deep learning means **many hidden layers**:

```text
Input  →  Hidden1  →  Hidden2  →  Hidden3  → ... →  Output
```

Why more layers matter:
- each layer learns a new representation
- later layers build on earlier learned features

---

### 2) But ML is Bigger than “Single Layer”

Many ML models are not defined by layers at all:

Examples:
- Decision Tree / Random Forest (tree splits)
- XGBoost (boosted trees)
- KNN (no training, uses distance)
- Naive Bayes (probability model)

So:
- DL = deep neural networks (layer-based)
- ML = broader category (shallow neural nets OR non-neural models)

---

### How Deep Learning Learns Features Automatically (Intuition + Math)

Each layer creates new features:

$$
z = Wx + b
$$

$$
a = f(z)
$$

Stacking layers:

$$
a^{(1)} = f(W^{(1)}x + b^{(1)})
$$

$$
a^{(2)} = f(W^{(2)}a^{(1)} + b^{(2)})
$$

So deeper layers can learn more complex patterns.

---

### Feature Learning Example (Easy)

#### In Images (CNN intuition)

- early layers learn edges and simple patterns
- deeper layers learn shapes and objects

#### In Text (Transformer intuition)

- model learns word meaning + context
- same word gets different representation in different sentences

---

### Why It Works

Loss tells the model what is wrong.

Gradients tell:
- which parameters caused the error
- how to change them to reduce the error

Repeated small updates:
- improve predictions
- build useful internal features automatically

