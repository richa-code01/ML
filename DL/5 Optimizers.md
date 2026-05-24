### 5) Optimizers (How weights are updated)

An **optimizer** is the rule that updates parameters to reduce loss.

General idea:
$$
	heta \leftarrow \theta - \alpha\,g_t
$$
where:
- $\theta$ = parameters (weights/bias)
- $\alpha$ = learning rate
- $g_t$ = gradient (or a transformed gradient) at step $t$

Most training uses **mini-batches**, so gradients are estimated from a batch.

---

### 0) SGD (Stochastic Gradient Descent)

In deep learning, “SGD” usually means **mini-batch SGD**.

Update:
$$
	heta_{t+1} = \theta_t - \alpha\,\nabla_{\theta} L(\theta_t)
$$

**Pros:**
- simple and memory-efficient
- often generalizes well

**Cons:**
- can be slow to converge
- sensitive to learning rate scheduling

**Where to use:**
- strong default for vision/CNNs (often with Momentum)
- when you care about generalization and can tune LR schedule

---

### 1) Momentum

Momentum accelerates SGD by adding a “velocity” term (reduces zig-zag).

Update:
$$
v_t = \beta v_{t-1} + (1-\beta)\,g_t
$$
$$
	heta_{t+1} = \theta_t - \alpha\,v_t
$$

Common value: $\beta=0.9$.

**Pros:**
- faster convergence than plain SGD
- smoother updates

**Cons:**
- extra hyperparameter ($\beta$)

**Where to use:**
- very common with SGD for CNNs / large-scale training

---

### 2) AdaGrad (Adaptive Gradient)

AdaGrad adapts learning rates per-parameter, giving bigger steps to rarely-updated parameters.

What “rarely-updated parameters” means:
- parameters whose gradients are **non-zero only on a small fraction of steps**, so they don’t get updated often
- this is common when inputs are **sparse** (many zeros), so most feature-related weights get gradient $0$ for most batches

Examples:
- **one-hot / bag-of-words / TF-IDF:** a rare word appears in few documents → its weight updates only when that word appears
- **embeddings:** a rare word/item embedding vector updates only when that token/item is in the mini-batch
- **recommenders:** user/item-specific parameters for rare users/items are updated infrequently

Update (element-wise):
$$
r_t = r_{t-1} + g_t^2
$$
$$
	heta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{r_t}+\varepsilon}\,g_t
$$

**Pros:**
- great for sparse features (NLP with one-hot/TF-IDF, some recommender systems)

**Cons:**
- learning rate keeps shrinking over time → can stop learning early
	- because $r_t$ is a cumulative sum, it keeps increasing
	- effective step size becomes $\frac{\alpha}{\sqrt{r_t}+\varepsilon}$, which keeps decreasing
	- after many steps, updates can become extremely small → training may look “stuck”

**Where to use:**
- sparse, convex-ish problems; less common for deep nets today

---

### 3) RMSProp

RMSProp fixes AdaGrad by using an exponential moving average of squared gradients.

Update (element-wise):
$$
s_t = \rho s_{t-1} + (1-\rho)g_t^2
$$
$$
	heta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{s_t}+\varepsilon}\,g_t
$$

Common values: $\rho=0.9$, $\varepsilon=10^{-8}$.

**Pros:**
- handles non-stationary objectives better than AdaGrad
- often works well for RNNs / noisy gradients

**Cons:**
- can still require LR tuning

**Where to use:**
- good practical choice; often replaced by Adam in many projects

---

### 4) Adam (Adaptive Moment Estimation)

Adam combines:
- Momentum (EMA of gradients)
- RMSProp idea (EMA of squared gradients)

Update (element-wise):
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

Bias-correction:
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t},\quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

Final update:
$$
	heta_{t+1} = \theta_t - \alpha\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}
$$

Common defaults:
- $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$

**Pros:**
- very strong “works out of the box” optimizer
- fast convergence
- good for noisy / sparse gradients

**Cons:**
- can generalize worse than well-tuned SGD+Momentum in some vision tasks
- weight decay handling matters (AdamW is preferred)

**Where to use:**
- default choice for many DL tasks (NLP/transformers, mixed workloads)
- when you want faster training with less LR-schedule complexity

---

### 5) AdamW (Adam + decoupled weight decay)

AdamW applies **weight decay** correctly by decoupling it from the gradient normalization.

Update idea:
$$
	heta \leftarrow \theta - \alpha\,\frac{\hat{m}}{\sqrt{\hat{v}}+\varepsilon} - \alpha\,\lambda\,\theta
$$
where $\lambda$ is weight decay.

**Pros:**
- usually better regularization behavior than Adam with L2-as-decay
- common default for transformers

**Cons:**
- still needs tuning of $\alpha$ and $\lambda$

**Where to use:**
- transformers / modern deep nets where weight decay is important

---

### Practical Tips

- **Learning rate is the most important hyperparameter** for almost all optimizers.
- If training is unstable (loss spikes / NaNs): try smaller $\alpha$, gradient clipping, or better initialization.
- If loss decreases but plateaus early: try LR schedule (decay), or switch optimizer (SGD↔Adam).

---

### Quick Recommendation

- If you want a safe default: **AdamW**
- If you want best generalization in vision (often): **SGD + Momentum** with a good LR schedule

