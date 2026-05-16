# Introduction to Machine Learning

# What is Machine Learning?

Machine Learning (ML) is a way of teaching computers to learn patterns from data so they can perform tasks on unseen data without explicitly programming every step.

Instead of manually writing rules, machines automatically discover patterns from data.

---

# Traditional Programming vs Machine Learning

## Traditional Programming

Input:
- Data
- Rules

Output:
- Result

```text
Data + Manual Logic → Result
```

---

## Machine Learning

Input:
- Data
- Expected Output

Output:
- Rules / Model

```text
Data + Expected Output → ML Algorithm → Model
```

The model learns patterns and uses them for future predictions.

---

# Types of Machine Learning based on the level of supervision required

# 1. Supervised Learning

## Definition

Supervised Learning uses labelled data where the model learns from input-output pairs.

Goal:
- Predict output for new unseen data

---

## Types of Supervised Learning

### Regression

Predicts continuous numerical values.

#### Examples
- House price prediction
- Temperature prediction

---

### Classification

Predicts categorical outputs.

#### Examples
- Spam Detection
- Disease Classification
- Pass / Fail Prediction

---

# 2. Unsupervised Learning

## Definition

Unsupervised Learning works on unlabelled data.

The model does not predict outputs but instead finds hidden patterns in data.

---

## Applications

### Clustering
Grouping similar data points together.

### Dimensionality Reduction
Reducing number of features while preserving information.

### Anomaly Detection
Finding unusual or abnormal data points.

### Association Rule Mining
Finding relationships between variables.

---

# 3. Semi-Supervised Learning

## Definition

Uses:
- Small amount of labelled data
- Large amount of unlabelled data

The model learns partially with supervision and partially by discovering patterns.
ex: google photos , we just give name to single pic of a person and then it groups all pics of same person in a single folder, simillarly does the same with multiple persons.

---

# 4. Reinforcement Learning

## Definition

Learning through:
- Rewards
- Penalties

Uses an agent-based approach.
No input data is given just scenarios and then based on its action it is rewarded or penalty and agent learns through that as it tries to maximize reward and minimize penalty.

---

## Applications

- Robotics
- Game AI
- Self-driving systems

---

# ML Training Methods

# Batch Learning

## Definition

Model is trained on the complete dataset at once, ususally done offline

---

## Advantages

- Stable learning

---

## Disadvantages

- Static model
- Requires retraining when data changes
- High hardware requirements

---

# Online Learning

## Definition

Model learns incrementally using small chunks of data, ususally done online i.e on server.

---

## Characteristics

- Learns continuously
- Updates itself regularly
- Suitable for streaming data

---

## Example

YouTube recommendation systems.

---

## Advantages

- Cost effective
- Handles continuously changing data

---

## Disadvantages

- Can learn from noisy or incorrect data

---

# Instance-Based Learning

## Definition

The model memorizes training data and compares new data with stored examples.

---

## Example

K-Nearest Neighbors (KNN)

---

# Model-Based Learning

## Definition

The model learns patterns and generalizes from data.

Instead of memorizing, it builds a mathematical representation.

---
