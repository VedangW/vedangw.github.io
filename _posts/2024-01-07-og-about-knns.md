---
layout: distill
title: About k-Nearest Neighbors
description: A deep dive into the specifics of k-nearest neighbors.
tags: data-mining, machine-learning, supervised-learning
giscus_comments: true
date: 2018-07-07
featured: false
published: true

authors:
  - name: Vedang Waradpande
    url: "https://vedangw.github.io/"
    affiliations:
      name: Birla Institute of Technology and Science, Pilani

bibliography: 2018-12-22-distill.bib

toc:
  - name: Introduction
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Algorithm
  - name: Choices
    subsections:
      - name: Distance Metric
      - name: Value of k
      - name: Vote Aggregation
  - name: Pros and Cons

---

## Introduction

While several complex algorithms have been developed for the purpose of solving classification and regression tasks in Machine Learning, there are cases in which simple algorithms work well and give satisfactory results with low (or no) training time. One such algorithm is the k-Nearest Neighbours algorithm. The idea of a k-NN is simple: "If it walks like a duck, quacks like a duck, then it is a duck."

Wikipedia gives a great definition for the k-Nearest Neighbours algorithm:
> "The k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space."

Desigining an effective k-NN classifier or regressor involves making a few choices according to the nature of the data. This article explores these choices and their effects on the algorithm's performance.

***

## Algorithm

k-Nearest Neighbours is a lazy evaluation algorithm, which means it doesn't have a training phase. It simply stores the training data and uses it to classify new instances. The inference phase is as follows.

The training data $$\mathbf{X}$$ is stored as a set of instances with input features $$X_1, ..., X_n$$ and corresponding labels $$Y$$. The label can be continuous (for regression) or discrete (for classification). A new sample $$X^* = (x_1, ..., x_n)$$ is represented using the same set of features and the goal is to predict $$y^*$$, the label.

Steps:

1. Calculate the pairwise distances $$d(X^i, X^*) \forall X^i \in \mathbf{X}$$ using a distance metric $$d$$.
2. Choose labels $$y^i$$ for the $$k$$ nearest neighbours of $$X^*$$.
3. The label is predicted as the mean of the labels of the $$k$$ nearest neighbours for continuous labels and as the mode for discrete labels.

***

## Choices

The process is quite straightforward, and in this section, we'll go through the choices to make.

### Distance Metric

To know which k instances are closest to the test instance, we'll need a distance metric $d$ to measure how different two instances are. There are several distance metrics which can be used depending upon the nature of the data and problem at hand.

- **Euclidean Distance**: The most commonly used distance metric. The Euclidean Distance between two points, $$A(a_1, a_2,..., a_n)$$ and $$B(b_1, b_2,..., b_n)$$ is given as 

$$
d(A, B) = \sqrt{\sum _{i=1}^{n}(a_i - b_i)^2}
$$

- **Manhattan Distance**: We can also use the Manhattan distance (also known as City Block distance). There are some claims that this metric works better for high dimensional data, as in [this paper](https://bib.dbvis.de/uploadedFiles/155.pdf). This distance is given by 

$$
d(A, B)=\sum _{i=1}^{n}\left|a_i - b_i\right|
$$

- **Hamming Distance**: In the case when $$X_1,...,X_n$$ are binary variables, the Hamming distance can be used. This is given by 

$$
\begin{align*}
  d(A, B) &= \sum_{i=1}^{n}I_i \text{, where} \\
  I_i &= \begin{cases} 
    1\text{, } a_i = b_i \\ 
    0\text{, otherwise}
  \end{cases}
\end{align*}
$$

### Value of k

This is the most important factor to consider and the only hyperparameter of the algorithm. How many nearest samples should we choose to make sure that a new instance gets classified correctly? Intuitively we can tell that the number shouldn't be too less or too many. Let's try to get an idea what happens when we choose different values for $k$.

1. **$$k$$ is too small**: Highly variable, unstable boundaries, sensitive to noise points and overfitting. Lets take $$k = 1$$ for a binary classification task (a 1-NN). This model will predict $$Y = 0$$ if the nearest neighbour to the test instance is negative and $$Y = 1$$ if it is positive. The "voting" fails because there aren't enough "voters".
2. **$$k$$ is too large**: Neighborhood may contain more instances from other classes, and thus the test instance can be easily misclassified. In case of a class imbalance and a large value of $k$, the predictions will tend to favour the majority class.
3. **$$k$$ is infinite**: This means that the k-NN always predicts the dominant class in the dataset. This is the same as a Zero-R rule-based classifier.


In most cases, there's no exactly right value of $k$ for a given dataset. One of the best ways to select $$k$$ is to hold out a set of points from the training set as a validation set and choosing the value which maximizes performance.

### Vote Aggregation

There are several ways to aggregate the votes (labels) of the $k$ nearest neighbours. The mode and mean are generally default options for discrete and continuous settings respectively. However, the votes can be weighted using several methods. Some examples are:

1. The weight is inversely proportional to the distances ($$w_i = \frac{1}{d(X^*, X^i)}$$). This means that the closer neighbours have a higher weight in the voting process. The weight can also be raised to an integer power to emphasize the effect of the distances.
2. In case there are timestamps associated with data points, weights can be assigned based on the recency of the data point. This is useful in cases where the data is non-stationary and the model needs to adapt to the changes in the data distribution.
3. In case of class imbalance, the weights can be assigned inversely proportional to the class frequencies. Lower frequency classes should have a bigger say in the voting process in many cases.

***

## Pros and Cons

The k-NN algorithm is a simple algorithm and can ideally be used as a baseline in research problems and for some simple software solutions. It's clearly not expressive enough to capture any complexity in real-world data, but following are some cases where it might be useful:

1. k-NN has a low setup times in terms of development and pairwise distances can be calculated fairly quickly by indexing the training data.
2. It's a lazy evaluation algorithm, so there's no training phase. This means that the model can be updated in real-time as new data comes in.
3. It's a non-parametric algorithm, which means there are no assumptions about the data distribution and the algorithm can be used universally to all kinds of data.
2. While rare in real-world data, the task can sometimes be simple and/or the data can already be neatly clustered using a preprocessing pipeline in higher dimensions. k-NN works very well in such cases.

It's also important to note the disadvantages of the algorithm:

1. The algorithm doesn't work in low data settings, and it might take a long time to gather enough data to make accurate predictions.
2. The algorithm is sensitive to noise and outliers. This can be mitigated to a certain extent by using a weighted voting scheme.
3. It can't capture non-linear relationships in the data.
4. However the distances are calculated, the algorithm is not scalable since it needs to calculate the pairwise distances between all the training instances and the test instance for each new instance and this can be computationally expensive.