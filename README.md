# KNN Iris Classification Project

## Overview
This project demonstrates the use of the K-Nearest Neighbors (KNN) algorithm to classify flowers in the Iris dataset.

The goal is to predict the species of a flower based on its features (sepal length, sepal width, petal length, petal width).

---

## Algorithm Explanation
KNN (K-Nearest Neighbors) is a simple and effective machine learning algorithm.

It works by:
- Finding the K closest data points to a new sample
- Assigning the class based on the majority vote

---

## Steps Performed
1. Load the Iris dataset  
2. Split the data into training and testing sets  
3. Train a KNN model  
4. Evaluate model accuracy  
5. Generate predictions  
6. Create a confusion matrix  
7. Plot accuracy vs K values  

---

## Results

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Accuracy vs K
![Accuracy vs K](accuracy_vs_k.png)

---

## How to Run
Make sure Python is installed, then run:

```bash
python iris_knn.py
