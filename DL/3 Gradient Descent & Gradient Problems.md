#### Gradient Descent update

$$
w := w - \alpha\frac{\partial L}{\partial w}
$$

$$
b := b - \alpha\frac{\partial L}{\partial b}
$$

where $\alpha$ is the learning rate.

---

### Gradient Descent (with curve intuition)

Gradient Descent is an **iterative optimization algorithm** that updates model parameters in the direction of the **negative gradient** to minimize the loss.

Negative gradient (brief):
- gradient $\nabla_\theta L$ points in the direction where loss increases fastest
- negative gradient $-\nabla_\theta L$ points in the direction where loss decreases fastest

Note:
- in non-convex losses (common in deep learning), it may converge to a **local** minimum / saddle point instead of the global minimum.

Goal of training:
- find parameters (weights/bias) that **minimize** the loss

$$
	heta^* = \arg\min_{\theta} L(\theta)
$$


Think of loss as a curve (for 1 parameter $w$):

```text
Loss L(w)
	^
	|        gradient < 0                 gradient > 0
	|       (left side)                   (right side)
	|            \                           /
	|             \                         /
	|              \                       /
	|               \                     /
	|                \                   /
	|                 \_____ w* ________/     → w
	|                       (minimum)
	|                    where dL/dw = 0
	+-------------------------------------------------> w

Left of minimum:
	dL/dw < 0  →  w := w - α(dL/dw) increases (moves right)

Right of minimum:
	dL/dw > 0  →  w := w - α(dL/dw) decreases (moves left)
```


Important note:
- this diagram shows a simple **convex** (bowl-shaped) loss curve, so there is a single minimum.
- in deep learning, $L(\theta)$ is usually **non-convex**, but gradient descent still works well in practice.

Meaning of the gradient (slope):
- $\frac{\partial L}{\partial w}$ tells how loss changes if you slightly change $w$
	- positive slope → increasing $w$ increases loss
	- negative slope → increasing $w$ decreases loss

Update rule (1D):

$$
w_{t+1} = w_t - \alpha\frac{\partial L}{\partial w}
$$

So:
- if slope is **positive**, subtracting a positive number moves $w$ left (downhill)
- if slope is **negative**, subtracting a negative number moves $w$ right (downhill)

In many parameters (vector form):
$$
	heta_{t+1} = \theta_t - \alpha\nabla_{\theta} L(\theta_t)
$$


Here $\nabla_{\theta} L$ is a vector pointing in the direction of **steepest increase**.
So moving in $-\nabla_{\theta} L$ takes the steepest step **downhill**.

---

### Vanishing Gradient Problem (Why it happens + how to overcome)

During backpropagation, gradients are passed from the output layer back to earlier layers.
In deep networks, these gradients can become **very small** as they move backward.
When gradients become near zero, early layers learn extremely slowly.

#### Why / how it happens

Backprop uses the chain rule, so the gradient becomes a product of many terms:

$$
\frac{\partial L}{\partial W^{(1)}} \approx \frac{\partial L}{\partial a^{(L)}}\;\prod_{l=2}^{L} \frac{\partial a^{(l)}}{\partial a^{(l-1)}}
$$

If many of these derivatives are numbers with magnitude less than 1, multiplying them repeatedly makes the product shrink exponentially.

Common reasons:

- **Sigmoid / tanh saturation:**
	- for large positive/negative inputs, sigmoid becomes almost flat
	- derivative becomes very small

For sigmoid:

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

Maximum value is 0.25 and often much smaller in saturation.

Why this causes vanishing gradients (chain multiplication):

- sigmoid output is always between $0$ and $1$
- if sigmoid output is near $0$ or $1$, then $\sigma'(z)$ becomes **very close to 0**

Even in the best case, sigmoid derivative is bounded:

$$
0 < \sigma'(z) \le 0.25
$$

During backprop, gradients multiply across layers (chain rule). If a network has many sigmoid layers, a simplified view is:

$$
\left|\frac{\partial L}{\partial W^{(1)}}\right| \propto \prod_{l=1}^{L} \sigma'\left(z^{(l)}\right)
$$

So the gradient magnitude is roughly bounded by:

$$
\prod_{l=1}^{L} \sigma'\left(z^{(l)}\right) \le (0.25)^L
$$

As $L$ grows, $(0.25)^L$ becomes extremely small, so early layers get almost zero gradient → **learning becomes very slow**.

- **Deep networks:** more layers → more multiplications → more shrinking
- **Bad initialization:** weights too small can shrink activations/gradients further

#### What you observe in training

- loss decreases very slowly
- early layers weights barely change
- model may fail to learn complex patterns

#### How to overcome vanishing gradients

- **Use ReLU family activations** (ReLU, Leaky ReLU, GELU)
	- they do not saturate the same way as sigmoid/tanh in the positive region

- **Good weight initialization**
	- Xavier/Glorot init (for tanh)
	- He initialization (for ReLU)

- **Batch Normalization**
	- stabilizes activations and helps gradient flow

- **Residual / skip connections** (ResNet idea)
	- provides alternate paths for gradients so they don’t vanish easily

- **For RNNs:** use **LSTM/GRU**
	- designed to reduce vanishing gradient issues in sequences

Note:

- vanishing gradients is different from **exploding gradients** (gradients become too large) — see next section.

---

### Exploding Gradient Problem (Why it happens + how to overcome)

Exploding gradients means gradients become **very large** during backprop.
This causes huge parameter updates and makes training unstable.

#### Why / how it happens

Similar to vanishing gradients, the chain rule multiplies many terms. But here, the product can become very large:

$$
\left|\prod_{l=2}^{L} \frac{\partial a^{(l)}}{\partial a^{(l-1)}}\right| \gg 1
$$

Common reasons:

- **Large weights / poor initialization:** can amplify activations and gradients
- **Deep networks:** more layers can amplify gradients if derivatives/weights lead to factors > 1
- **RNNs on long sequences:** repeated multiplication across many time steps can blow up gradients
- **High learning rate:** even if gradients are normal, too-large updates can make weights blow up, which then creates even larger gradients

#### What you observe in training

- loss becomes unstable (suddenly spikes)
- loss becomes `inf` or `NaN`
- weights become extremely large
- training accuracy fluctuates heavily

#### How to overcome exploding gradients

- **Gradient clipping (most common fix)**
	- clip gradients to a maximum norm/value before updating weights
	- prevents a single bad batch from destroying training

Simple idea:

$$
g := \nabla_\theta L,\quad g := \frac{g}{\max\left(1, \frac{\|g\|}{c}\right)}
$$

where $c$ is a chosen threshold.

- **Use a smaller learning rate**
	- reduces update size and prevents weight explosion

- **Better initialization** (Xavier/He)
	- keeps activations/gradients in a stable range

- **Batch Normalization / Residual connections**
	- helps stabilize training and gradient flow

- **For RNNs:** use **LSTM/GRU**
	- plus gradient clipping is very common in sequence models

---
