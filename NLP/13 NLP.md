### 13. NLP (Natural Language Processing)

NLP is a field of AI that helps machines:
- understand
- process
- generate

human language (text/speech).

In simple words:

```text
Human Language → Machine Understanding
```

Instead of:

```text
Only 0s and 1s
```

machines learn:
- English
- Hindi
- French
- etc.

---

### Real World Applications

- Chatbots & Virtual Assistants
- Gmail Spam Detection
- Sentiment Analysis
- Google Translate
- Voice Assistants
- Text Summarization
- Search Engines
- Auto Correct / Grammar Check
- Recommendation Systems
- Document Classification

Examples:
- Twitter sentiment analysis
- Amazon review analysis
- Customer feedback analysis

---

### Why NLP is Difficult?

Language is:

```text
Unstructured + Ambiguous
```

Problems:
- same word → multiple meanings (polysemy)
- different words → same meaning (synonyms)
- sarcasm/context understanding
- spelling mistakes
- emojis/slang
- grammar variations

Example:

```text
"Great, my laptop crashed again 😒"
```

Word:

```text
Great
```

looks positive,
but actual sentiment is negative.

---

### Approaches of NLP

---

#### 1. Rule-Based NLP (Old School NLP)

Idea:

```text
Manually write rules
```

System uses:
- grammar rules
- dictionaries
- handcrafted patterns

Examples:
- spell checker
- grammar checker
- regex matching
- keyword-based chatbot

Example Rule:

```text
IF sentence contains "buy now"
→ spam
```

---

#### Advantages

- easy for small systems
- interpretable
- deterministic

---

#### Limitations

- cannot handle ambiguity
- poor scalability
- difficult for large datasets
- fails on unseen patterns

Example:

```text
"That movie was sick!"
```

Rule-based system may think:

```text
"sick" = negative
```

but slang meaning could be positive.

---

#### 2. Statistical / ML-Based NLP

Language treated as:

```text
Data
```

ML models learn patterns from text.

Common models:
- Naive Bayes
- Logistic Regression
- SVM

Requires:

```text
Text → Numbers
```

using:
- BoW
- TF-IDF
- embeddings

---

#### Example

Train on:

```text
10,000 product reviews
```

Model learns:
- words like "good", "excellent" → positive
- words like "bad", "worst" → negative

Then predicts sentiment for new reviews.

Applications:
- spam detection
- sentiment analysis
- text classification

---

#### Limitations

- loses context
- poor semantic understanding
- sarcasm difficult

Example:

```text
"I absolutely love waiting 2 hours in traffic."
```

Model may think:

```text
love = positive
```

but actual meaning is sarcastic.

---

#### 3. Deep Learning NLP (Modern NLP)

Uses:

```text
Neural Networks
```

to automatically learn:
- patterns
- context
- semantics

Models:
- RNN
- LSTM
- GRU
- Transformers

Uses:
- embeddings
- attention
- contextual representations

Better understanding of:
- context
- sequence
- meaning

---

### NLP Pipeline

```text
1. Data Collection
2. Text Cleaning
3. Tokenization
4. Feature Extraction
5. Model Training
6. Evaluation
7. Deployment
```

---

### 1. Data Collection

Text can come from:
- Kaggle datasets
- PDFs
- text files
- websites
- APIs
- social media
- customer reviews
- emails
- surveys
- chats
- crowd sourcing
- web scraping

Libraries:
- BeautifulSoup (bs4)
- Scrapy
- Selenium

---

### 2. Text Cleaning / Preprocessing

Goal:

```text
Clean noisy text
```

---

#### Lowercasing

```text
"GOOD" → "good"
```

---

#### Remove Punctuation

```text
"hello!!!" → "hello"
```

In ML-based NLP:
- punctuation often removed

In DL NLP:
- punctuation may carry meaning

Example:

```text
"I won!"
"I won..."
```

---

#### Remove Numbers (Optional)

```text
"iphone 15" → "iphone"
```

depends on task.

---

#### Remove URLs

```text
https://abc.com
```

---

#### Remove HTML Tags

```html
<div> Hello </div>
```

---

#### Remove Emojis (Optional)

```text
😂🔥😍
```

May break vectorization in classical ML.

Modern transformers can use them.

---

#### Stopword Removal

Stopwords:

```text
is, the, was, in, and
```

Usually carry less meaning.

Example:

```text
"This is a book"
→ ["book"]
```

---

#### Spelling Correction (Optional)

```text
"gud" → "good"
```

---

### Tokenization

Token:

```text
Smallest unit of text
```

Examples:
- word
- character
- subword

Tokenization:

```text
Splitting text into tokens
```

Example:

```text
"I love NLP"
→ ["I", "love", "NLP"]
```

---

### Important Terms

#### Corpus

Complete collection of text data.

Example:

```text
All reviews dataset
```

---

#### Document

Single text sample.

Examples:
- one sentence
- one paragraph
- one review

---

#### Vocabulary (Vocab)

All unique words.

Example:

```text
[I, love, pizza, atul]
```

---

### Feature Extraction / Vectorization

ML models work on:

```text
Numbers
```

So text must be converted into vectors.

Methods:
1. One Hot Encoding
2. Bag of Words
3. TF-IDF
4. Word2Vec
5. BERT / Transformers

---

### Text Representation

Sentence/document converted into:

```text
Vector = list of numbers
```

Each number acts like:

```text
Feature
```

Example:

Vocabulary:

```text
[I, love, ML, NLP]
```

```text
"I love ML"
→ [1,1,1,0]
```

Matrix form:

```text
X = (documents × features)
```

called:

```text
Document-Term Matrix
```

---

## 1. One Hot Encoding (OHE)

Represents each word as:

```text
Binary Vector
```

Only one position is:

```text
1
```

others:

```text
0
```

---

### Example

Vocabulary:

```text
[atul, reads, books, richa, novels, library, has, both, and]
```

Vectors:

```text
atul   → [1,0,0,0,0,0,0,0,0]
reads  → [0,1,0,0,0,0,0,0,0]
books  → [0,0,1,0,0,0,0,0,0]
```

---

### Dataset Example

Documents:

```text
D1: "atul reads books"
D2: "richa reads novels"
D3: "library has both books and novels"
```

---

### One Hot Representation

```text
Vocab:
[atul, reads, books, richa, novels,
 library, has, both, and]

D1:
[1,1,1,0,0,0,0,0,0]

D2:
[0,1,0,1,1,0,0,0,0]

D3:
[0,0,1,0,1,1,1,1,1]
```

---

#### Advantages

- intuitive
- easy implementation

---

#### Limitations

##### Sparse Representation

Example:

```text
50k vocab size
```

Most values become:

```text
0
```

Huge memory waste.

---

##### Out of Vocabulary (OOV)

New unseen words ignored.

---

##### Semantic Meaning Not Captured

Example:

```text
"car" and "automobile"
```

treated completely unrelated.

---

##### Variable Length Problem

Different sentences:

```text
Different sizes
```

ML models usually need:

```text
Fixed-size vectors
```

---

## 2. Bag of Words (BoW)

Idea:

```text
Count frequency of words
```

Document represented using:

```text
(term, frequency)
```

---

### Example

Documents:

```text
Doc1: "I love ML"
Doc2: "I love NLP"
Doc3: "I love love NLP"
```

Vocabulary:

```text
[I, love, ML, NLP]
```

Vectors:

```text
Doc1 → [1,1,1,0]
Doc2 → [1,1,0,1]
Doc3 → [1,2,0,1]
```

---

### Similarity Intuition

More common words:

```text
→ higher similarity
```

Similarity methods:
- cosine similarity
- dot product

---

#### Advantages

- easy
- fixed size vectors
- works with ML
- fast baseline

---

#### Limitations

- sparse matrix
- ignores word order
- semantic meaning not captured
- OOV issue

---

## N-Grams

Previously:

```text
1 word = feature
```

called:

```text
Unigram
```

N-Gram:

```text
Sequence of N consecutive words
```

---

### Types

#### Unigram

```text
["I", "love", "NLP"]
```

---

#### Bigram

```text
["I love", "love NLP"]
```

---

#### Trigram

```text
["I love NLP"]
```

---

### Example

Sentence:

```text
"I am working in TRDDC Pune"
```

Bigrams:

```text
(I am)
(am working)
(working in)
(in TRDDC)
(TRDDC Pune)
```

---

### Why N-Grams Help?

Example:

```text
S1: "cricket is very good"
S2: "cricket is not good"
```

---

### Unigram Representation

Vocabulary:

```text
[cricket,is,very,not,good]
```

Vectors:

```text
S1 → [1,1,1,0,1]
S2 → [1,1,0,1,1]
```

Both look similar.

---

### Bigram Representation

Vocabulary:

```text
[cricket is,
 is very,
 very good,
 is not,
 not good]
```

Vectors:

```text
S1 → [1,1,1,0,0]
S2 → [1,0,0,1,1]
```

Now meanings become different.

---

#### Advantages

- captures local context
- better semantic meaning
- handles negation better

---

#### Limitations

- dimensions increase rapidly
- sparse matrix
- OOV issue

---

## 3. TF-IDF

Full Form:

```text
Term Frequency - Inverse Document Frequency
```

Idea:
- important words get high weight
- common words get low weight

---

### Term Frequency (TF)

Measures:

```text
How frequent word is inside document
```

Formula:

$$
TF(t,d)=\frac{count(t\ in\ d)}{total\ terms\ in\ d}
$$

---

### Inverse Document Frequency (IDF)

Measures:

```text
How rare word is across corpus
```

Formula:

$$
IDF(t)=\log\left(\frac{N}{df(t)}\right)
$$

Where:
- $begin:math:text$N$end:math:text$ = total documents
- $begin:math:text$df\(t\)$end:math:text$ = documents containing term

---

### TF-IDF Formula

$$
TFIDF(t,d)=TF(t,d)\times IDF(t)
$$

---

### Why "Inverse" Document Frequency?

Common words:

```text
the, is, and
```

appear everywhere.

So:

```text
Less useful
```

Rare words:

```text
cancer, transformer, bitcoin
```

carry more information.

Thus:

```text
Rare words → higher weight
```

---

#### Advantages

- better than BoW
- reduces importance of common words
- widely used in search engines
- works well with ML models

Used in:

```text
Google Search / Information Retrieval
```

---

#### Limitations

- sparse vectors
- OOV problem
- high dimensions
- weak semantic understanding

---

## 4. Word Embeddings

Represent words using:

```text
Dense Real-Valued Vectors
```

Instead of:

```text
Huge sparse vectors
```

---

### Core Idea

Words used in similar context:

```text
Have similar vectors
```

Example:

```text
king - man + woman ≈ queen
```

---

### Embedding Matrix

$$
E \in \mathbb{R}^{|V| \times d}
$$

Where:
- $begin:math:text$\|V\|$end:math:text$ = vocabulary size
- $begin:math:text$d$end:math:text$ = embedding dimension

---

### Why Better Than BoW?

- dense representation
- captures meaning
- semantic similarity learned
- smaller vectors

---

### Word2Vec

Learns embeddings using context.

---

#### CBOW

Predict center word using surrounding words.

---

#### Skip-Gram

Predict surrounding words using center word.

---

### GloVe

Uses:

```text
Global Word Co-occurrence
```

to learn embeddings.

---

### FastText

Uses:

```text
Character/Subword n-grams
```

Better for:
- rare words
- unseen words

---

#### Limitations

- static embeddings
- one word → one vector

Example:

```text
bank
```

same vector for:
- river bank
- finance bank

---

## 5. Contextual Embeddings

Meaning depends on:

```text
Context
```

Example:

```text
bank of river
bank account
```

Different embeddings generated.

---

### Based On

```text
Transformers
```

Models:
- BERT
- GPT
- RoBERTa

---

#### Advantages

- understands context
- handles polysemy
- state-of-the-art performance

---

#### Limitations

- computationally expensive
- large memory usage
- slower training

---

## Common NLP Tasks

### Text Classification

Examples:
- spam detection
- sentiment analysis

---

### Named Entity Recognition (NER)

Find entities:
- person
- location
- organization

Example:

```text
"Elon Musk lives in USA"

Person → Elon Musk
Location → USA
```

---

### Machine Translation

Example:

```text
English → French
```

---

### Text Summarization

#### Extractive

Pick important sentences.

#### Abstractive

Generate new summary.

---

## Common NLP Models

### Traditional ML

- Naive Bayes
- Logistic Regression
- SVM

---

### Deep Learning

- RNN
- LSTM
- GRU
- CNN
- Transformers

---

## Evaluation Metrics

Classification Metrics:
- Accuracy
- Precision
- Recall
- F1 Score

Confusion Matrix commonly used.

---

## Important Interview Points

- NLP converts text → numbers
- BoW/TF-IDF create sparse vectors
- embeddings create dense vectors
- TF-IDF used heavily in search systems
- transformers dominate modern NLP
- preprocessing depends on task
- stopword removal not always useful
- deep learning captures context better
