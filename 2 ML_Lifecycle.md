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
7. Test Model and optimize
        ↓
8. Deploy Model
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

Analyze patterns and relationships in data. trying to find hidden patterns ar associations in data.

---

## 5. Feature Engineering

Create useful features for better learning or drop some features if it is not contributing anything in the output.

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

## Pearson Correlation(used for numerical data)

Measures linear relationship between variables. here we can drop features having very low correlation with output variable.

---

## Chi-Square Test(used for categorical data)

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

Machine learning models work only with numerical data, so categorical values must be converted into numbers.

---

## Label Encoding

Assigns unique numbers to categories.
drawback: in distance based models high numeric value of a category can affect the output, thus we use this only when we have 2-3 categories for a particular feature.

### Example

```text
Red → 0
Blue → 1
Green → 2
```

---

## One-Hot Encoding

Creates separate binary columns for categories. It is widely used when multiple categories are there for a single feature.

| Color | Red | Blue | Green |
|---|---|---|---|
| Red | 1 | 0 | 0 |

---

# Feature Scaling

Feature scaling brings variables to the same scale.

---

## Min-Max Scaling also called normalization.

Transforms values between 0 and 1.

### Formula



x′ = (x − xₘᵢₙ) / (xₘₐₓ − xₘᵢₙ)

---

## Standardization (Z-Score Scaling)

Transforms data such that

- Mean = 0

- Standard deviation = 1
- ex:  transform values between -3 and 3

### Formula

z = (x − μ) / σ


Where:
- μ is mean of the data.
- σ is standard deviation of the data.

---
