### 4) Training Strategies (How Gradient Descent is applied)

When we say “Gradient Descent”, the core update is the same:

$$
	\theta \leftarrow \theta - \alpha\,\nabla_{\theta} L(\theta)
$$

But **how we estimate the gradient** depends on how many training examples we use per update.

Let the dataset be $\{(x_i, y_i)\}_{i=1}^{N}$ and loss per example be $\ell_i(\theta)=\ell(f_\theta(x_i),y_i)$.

---

### 1) Batch Gradient Descent (Full-batch GD)

Uses **all $N$ examples** to compute one gradient update.

Gradient estimate:
$$
\nabla_{\theta} L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\nabla_{\theta}\,\ell_i(\theta)
$$

**Pros:**
- stable and smooth updates (low variance)
- deterministic for a fixed dataset (same gradient each step)
- can converge nicely for convex problems / small models

**Cons:**
- very slow per update for large $N$ (each step scans full dataset)
- high memory/compute cost (not practical for big data)
- not ideal for non-stationary / streaming data

**Where / when to use:**
- small datasets where one full pass is cheap
- classical ML / convex optimization settings
- when you want very stable loss curves and can afford full-batch computation

---

### 2) Stochastic Gradient Descent (SGD)

Uses **one example** per update.

At step $t$, pick an example $i_t$ (often shuffled each epoch):
$$
	heta \leftarrow \theta - \alpha\,\nabla_{\theta}\,\ell_{i_t}(\theta)
$$

**Pros:**
- extremely fast updates (cheap step)
- works well for very large datasets
- noise in updates can help escape shallow local minima / saddle points (often helps generalization)
- can work in **online/streaming** learning (data arrives continuously)

**Cons:**
- noisy updates → loss curve fluctuates
- may not converge exactly to the minimum (often oscillates around it)
- sensitive to learning rate scheduling; may require more careful tuning

**Where / when to use:**
- online learning / streaming data
- very large datasets when you want quick incremental updates
- when you can tolerate noisy training and you’ll use learning-rate decay

Note:
- In deep learning, “SGD” is often used casually to mean **mini-batch SGD** (below), because pure single-example SGD is less common in practice.

---

### 3) Mini-batch Gradient Descent (Mini-batch GD)

Uses a **small batch of $B$ examples** per update (e.g., $B=16, 32, 64, 128$).

For a mini-batch $\mathcal{B}$ with $|\mathcal{B}|=B$:
$$
	heta \leftarrow \theta - \alpha\,\frac{1}{B}\sum_{i\in\mathcal{B}}\nabla_{\theta}\,\ell_i(\theta)
$$

**Pros:**
- balance between Batch GD (stable) and SGD (fast)
- efficient on GPUs/TPUs (vectorized computation)
- smoother than SGD but still has some helpful noise
- the most common and practical choice for deep learning

**Cons:**
- still requires choosing batch size $B$ (hyperparameter)
- very large batches can reduce the “noise benefit” and may generalize worse
- too small batches can be too noisy and slow on hardware

**Where / when to use:**
- almost always the default for deep learning training
- when training on GPUs/TPUs (best throughput)
- when you want a stable-but-efficient training process

Convergence note (mini-batch vs full-batch):
- in **wall-clock time**, mini-batch usually reaches a good loss/accuracy faster on large datasets (updates are much cheaper and GPU-friendly)
- in **number of updates**, full-batch can take fewer, smoother steps (it uses the exact dataset gradient), while mini-batch is noisier and may need more updates

To reach a good solution (time to good accuracy/loss):
- mini-batch typically wins for large datasets because you can make many quick updates and start improving early, while full-batch updates are expensive

Rule of thumb:
- small dataset → full-batch can be fine
- medium/large dataset + deep nets → mini-batch almost always converges faster in wall-clock time

---

### Quick Comparison (one line intuition)

- **Batch GD:** accurate gradient, slow step
- **SGD:** noisy gradient, very fast step
- **Mini-batch:** best tradeoff, fastest in practice on GPUs

---

### Pros/Cons Summary Table

| Method | Samples per update | Speed per update | Stability | Typical use |
|---|---:|---|---|---|
| Batch GD | $N$ | Slow | High | Small data / convex problems |
| SGD | $1$ | Very fast | Low (noisy) | Online learning / huge data |
| Mini-batch GD | $B$ | Fast (GPU-friendly) | Medium | Default for deep learning |

---

### Practical Tip (choosing batch size)

- If GPU memory is limited → reduce $B$
- If training is unstable/noisy → increase $B$ slightly (or lower learning rate)
- If throughput is low on GPU → try common values like $32, 64, 128$ and pick what fits memory

