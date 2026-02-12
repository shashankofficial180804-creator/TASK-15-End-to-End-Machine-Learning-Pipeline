# TASK-15-End-to-End-Machine-Learning-Pipeline
End-to-End Machine Learning Pipeline using Breast Cancer dataset. Includes preprocessing with ColumnTransformer, model training via Logistic Regression, evaluation metrics, and saved pipeline for deployment. Demonstrates production-ready ML workflow.

# Task 15 – End-to-End Machine Learning Pipeline

## 📌 Overview
This repository contains the implementation of **Task 15: End-to-End Machine Learning Pipeline** as part of the AI & ML Internship program.  
The objective of this task is to design, train, evaluate, and save a complete machine learning pipeline using best industry practices.

The project demonstrates how machine learning pipelines are used in real-world production systems to ensure consistency, prevent data leakage, and simplify deployment.

---

## 📊 Dataset
- **Primary Dataset:** Breast Cancer Dataset  
- **Source:** `sklearn.datasets`
- **Problem Type:** Binary Classification  
- **Classes:** Malignant (0) and Benign (1)

This dataset contains only numerical features, making it ideal for demonstrating preprocessing and pipeline construction.

---

## 🛠 Tools & Technologies
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
  - Pipeline  
  - ColumnTransformer  
  - StandardScaler  
  - Logistic Regression  
  - Evaluation Metrics  

---

## ⚙️ Project Workflow
The following steps were implemented as part of the end-to-end ML pipeline:

1. Loaded the Breast Cancer dataset and separated features and target variable.
2. Identified numerical features for preprocessing.
3. Applied feature scaling using `StandardScaler` via `ColumnTransformer`.
4. Built a complete machine learning pipeline combining preprocessing and model.
5. Split the dataset into training and testing sets (80-20 split).
6. Trained the pipeline on the training data.
7. Generated predictions on unseen test data.
8. Evaluated the model using multiple classification metrics.
9. Saved the trained pipeline as a `.pkl` file for deployment.

---

## 🔗 Machine Learning Pipeline
The pipeline includes:
- **Preprocessing:** Feature scaling using `StandardScaler`
- **Model:** Logistic Regression
- **Framework:** Scikit-learn `Pipeline` and `ColumnTransformer`

Using a pipeline ensures:
- No data leakage
- Consistent preprocessing during training and inference
- Easy deployment in production environments

---

## 📈 Model Evaluation Metrics
The trained model achieved the following performance on the test dataset:

- **Accuracy:** 0.9825  
- **Precision:** 0.9861  
- **Recall:** 0.9861  
- **F1 Score:** 0.9861  

These results indicate excellent classification performance and strong generalization.

---

## 💾 Saved Model
- **File Name:** `trained_pipeline.pkl`
- The saved file contains the complete pipeline (preprocessing + trained model).
- This allows direct loading and prediction on new data without reapplying preprocessing steps manually.

---

## 📁 Repository Structure
