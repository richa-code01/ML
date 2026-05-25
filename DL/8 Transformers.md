

# Transformers

## What are Transformers?

Transformers are deep learning architectures designed for:

- Natural Language Processing (NLP)
- Text generation
- Translation
- Chatbots
- Image understanding
- Speech processing

Transformers became revolutionary because they can process entire sequences in parallel.

Unlike RNNs, they do not process words one by one.

---

# Why Transformers Were Introduced

RNNs and LSTMs had problems:

- Slow sequential processing
- Difficulty handling long dependencies
- Vanishing gradient issues

Transformers solve this using:

- Attention mechanism
- Parallel processing
- Better context understanding

---

# Main Idea of Transformers

Instead of remembering information sequentially,

Transformers ask:

```text
Which words are important for understanding the current word?
```

This is done using Attention.

---

# Attention Intuition

Sentence:

```text
The animal didn't cross the street because it was tired.
```

To understand:

```text
it
```

the model learns that:

```text
it → animal
```

Attention helps the model focus on relevant words.

---

# Self-Attention

Self-attention means:

Every word looks at every other word in the sentence.

Example:

```text
I love deep learning
```

While processing:

```text
learning
```

the model checks:

- I
- love
- deep
- learning

and decides which words matter more.

---

# Core Components of Transformer

| Component | Purpose |
|---|---|
| Input Embedding | Converts words into vectors |
| Positional Encoding | Gives position information |
| Self-Attention | Learns word relationships |
| Feed Forward Network | Learns complex patterns |
| Softmax Layer | Produces probabilities |

---

# Input Embedding

Computers cannot understand text directly.

Words are converted into vectors.

Example:

```text
"cat" → [0.2, 0.8, 0.1, ...]
```

These numerical vectors capture semantic meaning.

---

# Why Positional Encoding is Needed

Transformers process words in parallel.

So they do not naturally know word order.

Example:

```text
Dog bites man
```

and

```text
Man bites dog
```

contain same words but different meanings.

Positional encoding gives order information.

---

# Self-Attention Mechanism

Each word creates:

| Vector | Purpose |
|---|---|
| Query (Q) | What am I searching for? |
| Key (K) | What information do I contain? |
| Value (V) | Actual information passed forward |

Attention score is calculated using Query and Key.

Higher score = more importance.

---

# Attention Formula

The scaled dot-product attention formula is:

genui{"math_block_widget_always_prefetch_v2":{"content":"\n\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V\n"}}

Where:

| Symbol | Meaning |
|---|---|
| Q | Query matrix |
| K | Key matrix |
| V | Value matrix |
| dₖ | Dimension of key vectors |

---

# Multi-Head Attention

Instead of using one attention mechanism,

Transformers use multiple attention heads.

Each head learns different relationships.

Example:

- One head learns grammar
- One head learns meaning
- One head learns long-term relationships

This improves understanding.

---

# Transformer Encoder

The encoder:

- Reads input sentence
- Extracts contextual understanding
- Produces rich representations

Used in models like:

- BERT

---

# Transformer Decoder

The decoder:

- Generates output word by word
- Uses encoder outputs
- Predicts next token

Used in models like:

- GPT

---

# Encoder-Decoder Flow

```text
Input Sentence
      ↓
Encoder
      ↓
Context Representation
      ↓
Decoder
      ↓
Output Sentence
```

---

# Popular Transformer Models

| Model | Purpose |
|---|---|
| BERT | Understanding text |
| GPT | Text generation |
| T5 | Text-to-text tasks |
| ViT | Vision Transformer for images |
| BART | Summarization and translation |

---

# BERT vs GPT

| Feature | BERT | GPT |
|---|---|---|
| Direction | Bidirectional | Left-to-right |
| Main Task | Understanding | Generation |
| Architecture | Encoder only | Decoder only |

---

# Why Transformers Became Powerful

Advantages:

- Parallel processing
- Handles long dependencies well
- Scales efficiently
- Better contextual understanding
- Works for text, images, audio, and video

---

# Limitation of Transformers

Transformers require:

- Huge datasets
- High computational power
- Large GPU memory

Training can be expensive.

---

# Applications of Transformers

| Application | Example |
|---|---|
| Chatbots | ChatGPT |
| Translation | Google Translate |
| Search Engines | Semantic search |
| Image Understanding | Vision Transformers |
| Speech Processing | Speech-to-text |
| Code Generation | GitHub Copilot |

---

# Difference Between RNN and Transformer

| Feature | RNN | Transformer |
|---|---|---|
| Processing | Sequential | Parallel |
| Memory | Hidden state | Attention |
| Long Dependencies | Difficult | Strong |
| Speed | Slower | Faster |

---

# Key Interview Points

## Why are transformers faster than RNNs?

Because they process all words in parallel.

---

## What is the heart of transformers?

Self-attention mechanism.

---

## Why is positional encoding needed?

Because transformers do not naturally understand sequence order.

---

## Difference between BERT and GPT?

- BERT understands text
- GPT generates text

---

# Quick Revision

| Concept | Key Idea |
|---|---|
| Transformer | Parallel sequence model |
| Attention | Focus on important words |
| Self-Attention | Words attend to each other |
| Positional Encoding | Adds word order |
| Multi-Head Attention | Multiple relationship learning |
| Encoder | Understands input |
| Decoder | Generates output |

---

# Final Intuition

Think of transformers like a giant discussion room 🧠

Every word can instantly talk to every other word.

Instead of remembering information step-by-step like RNNs,

Transformers build a global understanding of the entire sentence at once.

That ability changed modern AI forever.