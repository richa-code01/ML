# Convolutional Neural Networks (CNN)

## What is CNN?

A Convolutional Neural Network (CNN) is a type of deep learning neural network mainly used for:

- Image Classification
- Object Detection
- Face Recognition
- Medical Image Analysis
- Computer Vision Tasks

CNNs are specially designed to process grid-like data such as images.

---

# Why Not Use Simple Neural Networks for Images?

Suppose an image size is:

```text
1000 × 1000 × 3
```

Total input features:

```text
= 3,000,000
```

In a traditional ANN (Artificial Neural Network), input is taken as individual features.

For images, this means every pixel value is sent separately to the network.

So for an image with:

```text
3,000,000 pixels/features
```

millions of weights would be required even in the first hidden layer.

This makes ANN very inefficient for image processing.

Using fully connected layers would:

- Require huge memory
- Create too many parameters
- high computational power is required.
- Make training very slow
- looses spatial arrangement
- can lead to overfitting coz it memorizes pixel data

CNN solves this problem using:

- Local Connectivity
- Parameter Sharing
- Convolution Operations

---

# Basic Building Blocks of CNN

A CNN mainly consists of:

1. Convolution Layer
2. Activation Function
3. Pooling Layer
4. Fully Connected Layer

---


# Layers in CNN and Their Functions

| Layer | Purpose |
|---|---|
| Convolution Layer | Extracts important features like edges, textures, and patterns |
| Activation Layer (ReLU) | Introduces non-linearity so network can learn complex patterns |
| Pooling Layer | Reduces dimensions and computation while preserving important features |
| Flatten Layer | Converts 2D feature maps into 1D vector |
| Fully Connected Layer | Performs final classification or prediction |
| Output Layer | Produces final probabilities or class labels |

---

# How Each CNN Layer Performs Its Task

## 1. Convolution Layer

### What It Does

Extracts important visual features from the image. Multiple small filters/kernels slide over the image and extracts important features and each produce their own feature map. Pooling is applied independently on each feature map.

Examples:

- edges
- textures
- patterns
- shapes

---

### How It Works

A small filter/kernel slides over the image.

At every position:

1. Element-wise multiplication is performed
2. Values are summed
3. One output value is generated

This process creates a:

```text
Feature Map
```

Different kernels learn different patterns.

Example:

- One kernel may detect vertical edges
- Another may detect horizontal edges
- Another may detect textures

---

## 2. Activation Layer (ReLU)

### What It Does

Introduces non-linearity into the network.

Without activation functions:

- network behaves like simple linear model
- cannot learn complex patterns

---

### How It Works

ReLU applies:

```text
f(x) = max(0, x)
```

Meaning:

- Negative values become 0
- Positive values remain unchanged

Example:

```text
Input : [-2, 3, -1, 5]
Output: [ 0, 3,  0, 5]
```

This helps the network learn complex image relationships efficiently.

---

## 3. Pooling Layer

### What It Does

Reduces size of feature maps while keeping important information.

Benefits:

- reduces computation
- reduces overfitting
- speeds up training
- make model translation invariant i.e. position of object doesn't matter much.
- Avg. pooling is used when we want smoother, less noisy representations.
- Min pooling used for highlighting dark features.

---

### How It Works

Pooling scans small regions of feature map. It keeps only the important features while removing others.

Example using Max Pooling:

```text
[1 3]
[2 4]
```

Maximum value selected: Thus dominant features are selected.

```text
4
```

This reduces dimensions while preserving strongest features.

---

## 4. Flatten Layer

### What It Does

Converts 2D feature maps into 1D vector.

CNN layers produce:

```text
2D or 3D feature maps
```

But fully connected layers require:

```text
1D input vector
```

---

### How It Works

Example:

```text
2 × 2 feature map

[1 2]
[3 4]
```

After flattening:

```text
[1, 2, 3, 4]
```

---

## 5. Fully Connected Layer

### What It Does

Uses extracted features to perform classification.

It acts like a traditional ANN layer.

---

### How It Works

Each neuron is connected to all neurons from previous layer.

The layer:

1. Receives flattened features
2. Applies weights and biases
3. Learns relationships between features
4. Produces prediction scores

Example:

```text
Cat = 0.9
Dog = 0.1
```

---

## 6. Output Layer

### What It Does

Produces final prediction.

---

### How It Works

The output layer usually uses activation functions such as:

#### Sigmoid

Used for:

```text
Binary Classification
```

Example:

```text
Cat vs Dog
```

---

#### Softmax

Used for:

```text
Multi-class Classification
```

Example:

```text
Cat / Dog / Horse / Car
```

Softmax converts scores into probabilities.

Example:

```text
Cat   = 0.70
Dog   = 0.20
Horse = 0.07
Car   = 0.03
```

Highest probability becomes final prediction.

---

# How CNN Layers Work Together

## Early Layers

Learn simple features such as:

- edges
- lines
- corners

---

## Middle Layers

Learn more complex patterns such as:

- textures
- shapes
- object parts

---

## Deep Layers

Learn high-level concepts such as:

- faces
- animals
- objects
- complete structures

This hierarchical learning makes CNN extremely powerful for computer vision tasks.

---

## Kernel / Filter

A kernel is a small matrix.

Example:

```text
3 × 3 kernel
```

Example kernel:

```text
[ 1  0 -1 ]
[ 1  0 -1 ]
[ 1  0 -1 ]
```

This kernel helps detect vertical edges.

---

## Convolution Operation

The filter:

1. Multiplies element-wise with image pixels
2. Sums all values
3. Produces one output value
4. Slides to next position

This produces a:

```text
Feature Map
```

---

# Important Terms

## Stride

Stride determines how many steps the filter moves.

### Stride = 1

Filter moves one pixel at a time.

### Stride = 2

Filter skips one pixel.

Larger stride:

- Reduces output size
- Reduces computation

---

## Padding

Padding adds extra zeros around image boundaries.

### Why Padding?

Without padding:

- Output size shrinks
- Boundary information may be lost as boundary pixels are processed only once.

### Types of Padding

#### Valid Padding

No padding.

#### Same Padding

Padding added so output size remains same.

---

# Output Size Formula

For convolution:

```text
Output Size = ((N - F + 2P) / S) + 1
```

Where:

- N = Input size
- F = Filter size
- P = Padding
- S = Stride
---

# CNN Architecture Flow

- It can have multiple Convolution + Pooling layers for building understanding.

```text
Input Image
   ↓
Convolution
   ↓
ReLU
   ↓
Pooling
   ↓
Convolution
   ↓
ReLU
   ↓
Pooling
   ↓
Flatten
   ↓
Fully Connected Layer
   ↓
Output
```

---

# Why CNNs Work Well

CNNs automatically learn:

- Low-level features
  - edges
  - textures

- Mid-level features
  - shapes
  - object parts

- High-level features
  - complete objects

---

# Advantages of CNN

- High accuracy in image tasks
- Automatic feature extraction
- Fewer parameters than ANN
- Translation invariance

---

# Disadvantages of CNN

- Requires large dataset
- Computationally expensive
- Requires GPU for faster training
- Hard to interpret sometimes

---

# Popular CNN Architectures

## LeNet

Early CNN architecture.

Used for handwritten digit recognition.

---

## AlexNet

Won ImageNet competition in 2012.

Popularized deep learning in computer vision.

---

## VGGNet

Uses very small:

```text
3 × 3 filters
```

Deep but simple architecture.

---

## ResNet

Introduced:

```text
Skip Connections
```

Helps train very deep networks.

---

# Applications of CNN

- Self-driving cars
- Face recognition
- Medical diagnosis
- OCR
- Image classification
- Surveillance systems
- Satellite image analysis

---

# Difference Between ANN and CNN

| Feature | ANN | CNN |
|---|---|---|
| Input Handling | Flattened Input | Spatial Input |
| Best For | General Data | Images |
| Parameters | Huge | Fewer |
| Feature Extraction | Manual | Automatic |
| Computation | Expensive | Efficient |

---

# Important Interview Questions

## Why is ReLU used?

- Adds non-linearity
- Prevents vanishing gradients
- Computationally efficient

---

## Why is pooling used?

- Reduces dimensions
- Reduces overfitting
- Speeds up training

---

## What does a kernel do?

Kernel extracts important features like:

- edges
- textures
- patterns

---

## What is feature map?

Output generated after applying convolution.

---

## What is flattening?

Converting multidimensional feature maps into 1D vector before fully connected layers.

---

# Summary

CNN is a deep learning architecture mainly designed for image processing.

Key ideas:

- Convolution extracts features
- ReLU introduces non-linearity
- Pooling reduces dimensions
- Fully connected layers perform classification

CNNs are the backbone of modern computer vision systems.