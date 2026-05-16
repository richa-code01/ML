# Machine Learning Development Lifecycle

# Steps in ML Lifecycle

```text
1. Define Problem
        ↓
2. Gather Data
        ↓
3. Preprocess Data
        ↓
4. Perform EDA
        ↓
5. Feature Engineering
        ↓
6. Train Model
        ↓
7. Deploy Model
        ↓
8. Test & Optimize
```

---

# Step Explanations

## 1. Define Problem

Understand:
- Business problem
- Objectives
- Expected outcome

---

## 2. Gather Data

Collect required dataset from:
- Databases
- APIs
- CSV files
- Sensors
- Web scraping

---

## 3. Preprocess Data

Prepare data for ML model.

### Includes
- Cleaning
- Encoding
- Scaling
- Handling missing values

---

## 4. Perform EDA

Analyze patterns and relationships in data.

---

## 5. Feature Engineering

Create useful features for better learning.

---

## 6. Train Model

Train machine learning algorithm on prepared data.

---

## 7. Deploy Model

Deploy model into production/server.

---

## 8. Test & Optimize

Improve:
- Accuracy
- Performance
- Speed
- Generalization

---



# Feature Engineering and Feature Selection

# Feature Engineering

## Definition

Feature Engineering is the process of:
- Creating new features
- Transforming existing features

to expose useful patterns in the data.

---

# Feature Selection

## Definition

Selecting important features and removing irrelevant ones.

---

# Advantages

- Reduces complexity
- Improves performance
- Prevents overfitting

---

# Methods Used

## Correlation Analysis

Measures relationship between features and target variable.

---

## Pearson Correlation

Measures linear relationship between variables.

---

## Chi-Square Test

Used for categorical feature selection.

---


# EDA and Data Preprocessing

# Exploratory Data Analysis (EDA)

## Definition

EDA is the process of exploring and understanding data before training machine learning models.

---

# Goals of EDA

- Understand dataset structure
- Discover patterns
- Detect anomalies
- Identify missing values

---

# Common EDA Operations

## View Shape

```python
data.shape
```

---

## View First Rows

```python
data.head()
```

---

## Summary Statistics

- Mean
- Median
- Unique values
- Null values

---

# Data Cleaning

Data cleaning improves dataset quality.

---

# Tasks in Data Cleaning

- Handling missing values
- Removing wrong values
- Detecting outliers
- Fixing inconsistencies

---

# Encoding Categorical Data

Machine learning models work with numerical data, so categorical values must be converted into numbers.

---

## Label Encoding

Assigns unique numbers to categories.

### Example

```text
Red → 0
Blue → 1
Green → 2
```

---

## One-Hot Encoding

Creates separate binary columns for categories.

| Color | Red | Blue | Green |
|---|---|---|---|
| Red | 1 | 0 | 0 |

---

# Feature Scaling

Feature scaling brings variables to the same scale.

---

## Min-Max Scaling

Transforms values between 0 and 1.

### Formula

$begin:math:display$
x\' \= \\frac\{x \- x\_\{min\}\}\{x\_\{max\} \- x\_\{min\}\}
$end:math:display$

---

## Standardization (Z-Score Scaling)

Transforms data such that:
- Mean = 0
- Standard deviation = 1

### Formula

$begin:math:display$
z \= \\frac\{x \- \\mu\}\{\\sigma\}
$end:math:display$

Where:
- $begin:math:text$\\mu$end:math:text$ = Mean
- $begin:math:text$\\sigma$end:math:text$ = Standard deviation

---
