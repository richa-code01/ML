### 4. Decision Tree

A tree-based supervised learning algorithm used for:
- Classification
- Regression

Makes predictions using decision conditions.

---

### Example

```text
if money < 30      → Snack
if 30 < money < 60 → Fast Food
if 60 < money < 90 → Dinner
if money > 90      → 5-Star Hotel
```

Decision Tree converts such rules into a tree structure.

---

### Tree Components

- Root Node → first/best split feature
- Non-Leaf Node → decision conditions
- Leaf Node → final output/class
- Branch → outcome of a condition

---

### Play Tennis Example

#### Root Node → Outlook

```text
Outlook
│
├── Sunny     → (2 Yes, 3 No)
├── Overcast  → (4 Yes, 0 No) → Play = Yes
└── Rain      → (3 Yes, 2 No)
```

`Overcast` is already pure:

```text
Only YES present
```

So it directly becomes a leaf node.

---

### Expanding Sunny Branch

```text
Sunny
│
└── Temperature
     │
     ├── Hot  → (0 Yes, 2 No) → Play = No
     ├── Mild → Mixed
     └── Cool → Mixed
```

Tree recursively keeps splitting data.

---

### Final Decision Tree

```text
                Outlook
          /        |        \
      Sunny    Overcast     Rain
        |          |           |
   Temperature    Yes       Wind
    /   |   \                 / \
 Hot Mild Cool            Weak Strong
  No   Yes  Yes            Yes   No
```

---

### Working of Decision Tree

#### Steps

1. Start with dataset
2. Calculate entropy for each feature
3. Find Information Gain
4. Choose feature with highest Information Gain
5. Split dataset into branches
6. Repeat recursively
7. Stop when:
   - node becomes pure
   - max depth reached
   - no features left

---

### Entropy

Measures impurity/disorder in dataset.

#### Formula

$$
Entropy(S)=-p_+\log_2(p_+)-p_-\log_2(p_-)
$$

Where:
- `p+` → probability of YES
- `p-` → probability of NO

---

### Entropy Intuition

| Situation | Entropy |
|---|---|
| All Yes / All No | 0 (Pure) |
| 50% Yes, 50% No | 1 (Maximum impurity) |

- Lower entropy → cleaner split
- Higher entropy → more mixed data

---

### Small Numerical Example

Dataset:

```text
Yes = 3
No  = 1
```

Total:

```text
4
```

Probabilities:

$$
p_+=\frac{3}{4}=0.75
$$

$$
p_-=\frac{1}{4}=0.25
$$

Entropy:

$$
Entropy(S)=-(0.75)\log_2(0.75)-(0.25)\log_2(0.25)
$$

$$
Entropy(S)\approx0.81
$$

---

### Information Gain

Measures reduction in entropy after split.

Feature with highest Information Gain becomes root node.

#### Formula

$$
IG(S,Feature)=Entropy(S)-\sum \frac{|S_v|}{|S|}\times Entropy(S_v)
$$

Where:
- `S` → complete dataset
- `Sv` → subset after split
- `|Sv|` → subset size
- `|S|` → total dataset size

---

### Information Gain Intuition

- High Information Gain → better feature
- Goal:
  - reduce entropy
  - create pure nodes

```text
Higher IG = Better Split
```

---

### Overfitting in Decision Trees

Decision Trees can memorize training data.

#### Solution

- Max Depth
- Pruning
- Minimum Samples Split

---

### Important Interview Points

- No feature scaling required
- Handles categorical + numerical data
- Can model non-linear boundaries
- Sensitive to noisy datasets
- Greedy algorithm:
  - chooses locally best split each time

---

### Advantages

- Easy to understand & visualize
- Fast training
- Works well on non-linear data
- Interpretable model

---

### Disadvantages

- Overfitting problem
- Unstable with noisy data
- Small data change can change entire tree
