### 13. NLP (Natural Language Processing)

NLP is a field of AI that helps machines understand, process and generate **human language** (text/speech).

Examples:
- Spam detection
- Sentiment analysis
- Machine translation
- Chatbots
- Document classification

---

### Why NLP is Difficult?

- same word can have different meanings (polysemy)
- different words can have same meaning (synonyms)
- grammar + context matter
- spelling mistakes, slang, emojis
- language is unstructured

---

### NLP Pipeline (High Level)

1. Text collection
2. Text cleaning + preprocessing
3. Convert text → numbers (vectorization/embeddings)
4. Train ML/DL model
5. Evaluate + deploy

---

### Text Preprocessing

Common steps (depends on task):

#### 1) Lowercasing

```text
"GOOD" → "good"
```

#### 2) Remove punctuation / special symbols

```text
"hi!!!" → "hi"
```

#### 3) Tokenization

Split text into words/tokens.

Example:

```text
"I love ML" → ["I", "love", "ML"]
```

#### 4) Stopword Removal

Remove very common words that may not add meaning.

Example:

```text
"this is a book" → ["book"]
```

#### 5) Stemming

Convert words to root form (may not be a real word).

Example:

```text
"playing" → "play"
"studies" → "studi"
```

#### 6) Lemmatization

Convert words to dictionary form (meaningful base word).

Example:

```text
"better" → "good"
"running" → "run"
```

> Lemmatization is usually better than stemming but slower.

---

### Text Representation (Text → Numbers)

ML models work on numbers, so we convert text into vectors.

---

### 1) Bag of Words (BoW)

Idea:
- create vocabulary of all words
- represent each document by word counts

Example:

```text
Doc1: "I love ML"
Doc2: "I love NLP"

Vocab: [I, love, ML, NLP]
Doc1: [1, 1, 1, 0]
Doc2: [1, 1, 0, 1]
```

Limitation:
- ignores word order
- does not capture meaning

---

### 2) TF-IDF

TF-IDF gives higher weight to words that are:
- frequent in a document
- but rare across all documents

Formulas:

Term Frequency (TF):

$$
TF(t,d)=\frac{count(t\ in\ d)}{total\ terms\ in\ d}
$$

Inverse Document Frequency (IDF):

$$
IDF(t)=\log\left(\frac{N}{df(t)}\right)
$$

TF-IDF:

$$
TF\text{-}IDF(t,d)=TF(t,d)\times IDF(t)
$$

---

### 3) Word Embeddings

Embeddings represent words as dense vectors (captures meaning).

Common embeddings:
- Word2Vec (CBOW, Skip-gram)
- GloVe
- FastText

Properties:
- similar words have similar vectors
- captures relationships

Example idea:

```text
vec("king") - vec("man") + vec("woman") ≈ vec("queen")
```

---

### 4) Contextual Embeddings

Same word can have different meaning in different sentences.

Example:

```text
"bank" (river bank)
"bank" (money bank)
```

Contextual models:
- BERT
- GPT
- RoBERTa

They generate embeddings depending on the context.

---

### Common NLP Tasks

#### 1) Text Classification

Examples:
- spam vs not spam
- positive vs negative sentiment

#### 2) Named Entity Recognition (NER)

Find entities:
- Person, Location, Organization

Example:

```text
"Elon Musk lives in USA"
Person: Elon Musk
Location: USA
```

#### 3) Machine Translation

English → French, etc.

#### 4) Text Summarization

- extractive
- abstractive

---

### Common Models Used in NLP

Traditional ML:
- Naive Bayes (good baseline for text)
- Logistic Regression
- SVM

Deep Learning:
- RNN / LSTM / GRU
- CNN for text
- Transformers (BERT, GPT)

---

### Evaluation Metrics

For classification:
- Accuracy
- Precision, Recall
- F1-score

Confusion matrix is often used.

---

### Important Notes

- Text data is sparse in BoW/TF-IDF
- Feature scaling is not used the same way for text vectors
- More data usually improves NLP models
- Preprocessing depends on the task (don’t remove useful words)

