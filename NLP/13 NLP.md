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

### Text Representation (Text → Numbers) also called vectorization

ML models work on numbers, so we convert each text (sentence/document) into a **vector**. sentence is also called document.

Meaning:
- a vector is just a list of numbers
- each number behaves like a **feature** for the ML model

Example (Bag of Words / TF-IDF):
- suppose vocabulary = `[I, love, ML, NLP]`
- then each sentence becomes a fixed-size vector of length 4

```text
"I love ML"  → [1, 1, 1, 0]
"I love NLP" → [1, 1, 0, 1]
```

So the ML model sees input like:

```text
X = (number of documents) × (number of features)
```

This is also called a **document-term matrix**.

Important:
- ML models usually need **fixed-size** input vectors
- different vectorization methods create different types of features
	- counts (BoW)
	- weighted counts (TF-IDF)
	- dense semantic vectors (embeddings)

---

### 1) One-Hot Encoding

One-hot encoding represents each word as a binary vector.

Idea:
- create a vocabulary of all unique words
- each word is represented by a vector of length = vocabulary size
- only **one position is 1**, all others are 0

Example:

```text
Vocab: [I, love, ML, NLP]

I   → [1, 0, 0, 0]
ML  → [0, 0, 1, 0]
NLP → [0, 0, 0, 1]
```

How to represent a full sentence/document:
- as a sequence of one-hot vectors (token by token), OR
- by summing/counting them (this becomes Bag of Words)

Limitations:
- very high-dimensional and sparse (too many zeros)
- does not capture meaning ("ML" and "AI" are unrelated)
- vocabulary is built during training; if a new word (out-of-vocabulary) appears in testing, it may be ignored → information loss
- if we keep it as a sequence, sentences can have different lengths → ML models usually need fixed-size input

---

### 2) Bag of Words (BoW)

Idea:
- create vocabulary of all words
- represent each document by word counts
- similarity can be computed using vector similarity (dot product / cosine similarity)

Simple intuition:
- two sentences are more similar if they share more common words (with similar counts)

Example:

```text
Doc1: "I love ML"
Doc2: "I love NLP"
Doc3: "I love love love NLP"

Vocab: [I, love, ML, NLP]
Doc1: [1, 1, 1, 0]
Doc2: [1, 1, 0, 1]
Doc3: [1, 3, 0, 1]
```
Pros:
- creates fixed size vectors for each sentence(input).
- simple and fast baseline

Limitation:
- ignores word order
- does not capture meaning
- sparse matrix
- out of vocabulary issue.

---

### 3) TF-IDF

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

#### N-grams in TF-IDF

So far we assumed each **feature** is a single word (this is called a **unigram**).

**N-gram** = a sequence of $n$ consecutive tokens.

- **1-gram (unigram):** `"not"`, `"good"`
- **2-gram (bigram):** `"not good"`, `"machine learning"`
- **3-gram (trigram):** `"I do not"`

Example:

```text
Sentence: "this movie is not good"
Unigrams : [this, movie, is, not, good]
Bigrams  : [this movie, movie is, is not, not good]
```

Why bigrams can beat unigrams (simple example):

```text
S1: "cricket is not good"
S2: "cricket is very good"

Unigram vocab  : [cricket, is, not, very, good]
S1 unigram vec : [1, 1, 1, 0, 1]
S2 unigram vec : [1, 1, 0, 1, 1]

These vectors share most words, so they look very similar,thus cosine similarity will be high even though the meanings are opposite.

Bigram vocab   : [cricket is, is not, not good, is very, very good]
S1 bigram vec  : [1, 1, 1, 0, 0]
S2 bigram vec  : [1, 0, 0, 1, 1]

Now the key phrases "not good" vs "very good" become different features and the vectors will have low similarity as expected.
```

When we use TF-IDF with n-grams, the **vocabulary** can contain unigrams + bigrams + trigrams, and the document-term matrix becomes larger.

How it helps:
- captures short phrases / local word-order information
- handles negation better (e.g., `"not good"` is different from `"good"`)
- improves classification tasks where phrases matter

Trade-offs:
- vocabulary size grows fast → more memory and slower training
- can overfit if you use high $n$ on small datasets

When to use n-grams (practical guidance):
- Use **unigram TF-IDF** as a strong baseline for many text classification tasks.
- Add **bigrams** when meaning depends on phrases (sentiment, spam, intent, topic labels).
- Use **trigrams** only if you have enough data and phrases of length 3 matter.

Also: **Word n-grams vs Character n-grams**

- **Word n-grams** (default): best when tokens are clean and you care about phrases.
- **Character n-grams**: helpful for noisy text (typos, usernames, hashtags), and for languages with rich morphology.

Common choices:
- word n-grams: `(1,2)` (unigram + bigram)
- char n-grams: `(3,5)` (character 3 to 5-grams)

---

### 4) Word Embeddings

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

### 5) Contextual Embeddings

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

