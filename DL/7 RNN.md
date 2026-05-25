

# Recurrent Neural Network (RNN)

## What is an RNN?

A Recurrent Neural Network (RNN) is a type of neural network designed for:

- Sequential data -> data where sequence of input matters like sentence.
- Time-series data
- Language processing
- Speech recognition
- Text prediction

Unlike traditional neural networks, RNNs have memory.

They remember previous information using a hidden state.

---

# Why Normal Neural Networks Fail for Sequential Data

A traditional neural network treats every input independently.

Example:

```text
I love deep learning
```

To understand the word:

```text
learning
```

the model should remember:

```text
I love deep
```

A normal neural network cannot remember previous words.

RNN solves this using feedback connections.

---

# Basic Structure of RNN

At every timestep:

```text
Current Input + Previous Memory → New Memory + Output
```

The hidden state acts as memory.

---

# RNN Flow

```text
x₁ → h₁ → y₁
      ↓
x₂ → h₂ → y₂
      ↓
x₃ → h₃ → y₃
```

Where:

| Symbol | Meaning |
|---|---|
| xₜ | Input at timestep t |
| hₜ | Hidden state (memory) |
| yₜ | Output |

---

# Hidden State Update

The hidden state is updated using:

```text
hₜ = f(Wxₜ + Uhₜ₋₁)
```

Where:

| Term | Meaning |
|---|---|
| xₜ | Current input |
| hₜ₋₁ | Previous hidden state |
| W, U | Weight matrices |
| f | Activation function |

---

# Important Idea

RNN reuses the same weights at every timestep.

This helps:

- Reduce parameters
- Learn sequence patterns
- Handle variable-length inputs

---

# Example: Sentence Processing

Sentence:

```text
I love AI
```

Processing happens sequentially:

```text
Step 1:
Input = I
Memory updated

Step 2:
Input = love
Uses previous memory
Memory updated

Step 3:
Input = AI
Uses previous memory
Final output generated
```

The model gradually builds contextual understanding.

---

# Applications of RNN

| Application | Example |
|---|---|
| Language Translation | English → French |
| Text Prediction | Next word suggestion |
| Sentiment Analysis | Positive/Negative review |
| Speech Recognition | Voice assistants |
| Time Series Forecasting | Stock prediction |

---

# Types of RNN Architectures

## One-to-One

```text
Image → Label
```

Example:

- Image classification

---

## One-to-Many

```text
Image → Caption Sentence
```

Example:

- Image caption generation

---

## Many-to-One

```text
Words → Sentiment
```

Example:

- Sentiment analysis

---

## Many-to-Many

```text
English Sentence → French Sentence
```

Example:

- Machine translation

---

# Problem with Basic RNN

RNN suffers from:

- Vanishing gradient problem
- Difficulty remembering long-term dependencies

Example:

```text
The movie released last year was fantastic because ...
```

The model may forget earlier important words.

---

# Vanishing Gradient Problem

During backpropagation:

- Gradients become extremely small
- Early layers learn very slowly
- Long-term memory becomes weak

This makes training difficult for long sequences.

---

# Solution: LSTM and GRU

Advanced RNN variants:

| Model | Purpose |
|---|---|
| LSTM | Better long-term memory |
| GRU | Simpler and faster version of LSTM |

These solve memory-related issues.

---

# LSTM (Long Short-Term Memory)

LSTM introduces:

- Forget gate
- Input gate
- Output gate

These gates control:

- What to remember
- What to forget
- What to output

LSTM acts like a smart memory manager.

---

# GRU (Gated Recurrent Unit)

GRU is a simplified version of LSTM.

It uses:

- Reset gate
- Update gate

Advantages:

- Faster training
- Fewer parameters
- Good performance

---

# Difference Between ANN, CNN, and RNN

| Network | Best For |
|---|---|
| ANN | General tabular data |
| CNN | Images and spatial data |
| RNN | Sequential and time-series data |

---

# Key Interview Points

## Why RNN is called recurrent?

Because the output from previous timestep is fed back into the network.

---

## What gives memory to RNN?

The hidden state.

---

## Main limitation of RNN?

Vanishing gradient problem.

---

## Which models improve RNN?

- LSTM
- GRU

---

# Quick Revision

| Concept | Key Idea |
|---|---|
| RNN | Neural network for sequences |
| Hidden State | Memory of previous inputs |
| Sequential Processing | Inputs processed one after another |
| Vanishing Gradient | Difficulty learning long dependencies |
| LSTM/GRU | Improved memory handling |

---

# Final Intuition

Think of RNN like reading a story 📖

While reading each new word:

- you remember previous words
- build context continuously
- use past information to understand the future

That rolling memory is the heart of RNNs.