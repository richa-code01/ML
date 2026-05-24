### 4) Common Activation Functions

#### Linear / Identity (mostly for regression output)

Linear activation means the neuron outputs the same value it receives:

$$
f(x) = x
$$

When it is used:
- mainly in the **output layer** for **regression** (predict any real value like price, temperature)

Where to use:
- **output layer** for **regression** tasks (predict any real value)

Why not used in hidden layers:
- it does not add non-linearity
- if all layers use linear activation, then even a deep network becomes equivalent to a single linear model

#### Sigmoid (binary classification output)

- S shaped curve that squashes value between 0 and 1.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Output range: $[0, 1]$ (probability)

Where to use:
- **output layer** for **binary classification** (predict probability of class 1)
- when you specifically need a probability-like output between 0 and 1

When not to use:
- usually avoid in **hidden layers** of deep networks
	- can cause **vanishing gradients** due to saturation
	- training becomes slow compared to ReLU-family activations

#### Tanh (often better than sigmoid, but still can vanish)

Tanh is also an S-shaped function, but it is **zero-centered** (outputs can be negative and positive).

$$
	anh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

Output range: $[-1, 1]$

Where to use:
- can be used in some **hidden layers** (especially in older networks)
- commonly used inside **RNN/LSTM** components (gates use sigmoid/tanh together)

When not to use:
- very deep feed-forward networks (many layers)
	- tanh can still **saturate** for large $|x|$ → gradients become small → vanishing gradients
- if ReLU-family activations work well for your task (usually the default choice)

#### Softmax (multi-class classification output)

$$
softmax(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

Probability note:
- softmax converts logits into values in $[0,1]$ that sum to 1, so it can be interpreted as the **probability of each class** (for **single-label** multi-class classification)
- for **multi-label** classification (multiple classes can be true), use **sigmoid** independently for each class instead of softmax

Output layer note:
- for $K$-class (single-label) classification, the output layer usually has **$K$ neurons** (one logit per class), followed by softmax
- binary classification is commonly done with **1 output neuron + sigmoid** (or sometimes 2 neurons + softmax)

Where to use:
- **output layer** for **multi-class, single-label** classification (choose 1 class out of $K$)
- typically paired with **cross-entropy loss**

When not to use:
- hidden layers (it forces activations to sum to 1)
- multi-label classification (use **sigmoid** per class instead)

#### ReLU (Rectified Linear Unit - most common in hidden layers)

$$
ReLU(x) = \max(0, x)
$$

Where to use:
- default choice for **hidden layers** in many feed-forward networks and CNNs
- when you want a simple, fast non-linearity and stable gradients (vs sigmoid/tanh)

Pros:
- fast
- reduces vanishing gradient compared to sigmoid/tanh

Limitations:
- **Dying ReLU problem:** for negative inputs, gradient becomes 0 (neuron can get stuck outputting 0)
- not **zero-centered** (outputs are non-negative), sometimes slows optimization
- output is **unbounded** for positive values (can lead to large activations if not controlled)

#### Leaky ReLU (fix for dying ReLU)

Leaky ReLU is a small modification of ReLU where we allow a **small negative slope** instead of making all negative values exactly 0.

$$
LeakyReLU(x) = \max(\alpha x, x)
$$

Equivalently:
$$
LeakyReLU(x) =
\begin{cases}
x & x \ge 0 \\
\alpha x & x < 0
\end{cases}
$$

where $\alpha$ is a small constant (commonly $0.01$).

Where to use:
- hidden layers when plain ReLU shows **dying ReLU** (many neurons stuck at 0)
- as a drop-in replacement for ReLU if you want a small gradient for $x<0$

Why it is used:
- reduces the **dying ReLU** issue (negative side still has gradient $\alpha$ this way, the neuron is never completely dead.)
- often trains more stably than plain ReLU in some setups

Limitations:
- still **unbounded** for positive values (can lead to large activations if not controlled)
- you must choose $\alpha$ (extra hyperparameter; too small behaves like ReLU, too large can hurt learning)
- not always better than ReLU (performance can be task- and initialization-dependent)

#### PReLU (Parametric ReLU)

PReLU is like Leaky ReLU, but the negative slope is **learned from data** instead of being a fixed constant.

$$
PReLU(x) = \max(\alpha x, x)
$$

Equivalently:
$$
PReLU(x) =
\begin{cases}
x & x \ge 0 \\
\alpha x & x < 0
\end{cases}
$$

where $\alpha$ is a **trainable parameter** (it can be one per neuron, or one per channel in CNNs).

Where to use:
- hidden layers when you want the model to **learn** the negative slope instead of choosing it manually
- commonly used in some CNN architectures (often with one $\alpha$ per channel)

Why it is used:
- keeps a non-zero gradient for $x<0$ (helps with **dying ReLU**)
- model can adapt the negative slope automatically (sometimes improves accuracy)

Limitations:
- adds extra parameters (small, but can increase overfitting risk)
- learned $\alpha$ can become too large in some cases (may hurt stability if not regularized/controlled)

#### Swish (modern smooth activation)

Swish is a smooth, non-linear activation defined as:

$$
Swish(x) = x \cdot \sigma(\beta x)
$$

Most commonly, $\beta = 1$, so:
$$
Swish(x) = x \cdot \sigma(x)
$$

Intuition:
- for large positive $x$, $\sigma(x) \approx 1$ so $Swish(x) \approx x$ (behaves like linear on positives)
- for negative $x$, $\sigma(x)$ is small, so output is suppressed but not hard-zero (unlike ReLU)

Where to use:
- hidden layers in deeper networks where you want a smooth activation (often used in modern CNN families)
- when you can afford a bit more compute than ReLU

Advantages:
- smooth gradients (often helps optimization)
- does not hard-zero all negative inputs (can reduce “dead neuron” behavior compared to ReLU)
- can outperform ReLU on some deep networks (commonly seen in modern CNN families)

Limitations:
- slightly more compute than ReLU (needs sigmoid)
- not guaranteed to be better than ReLU; depends on architecture and data
- can saturate for very negative inputs (gradients can still become small there)

---
