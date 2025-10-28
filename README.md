# 💧 Water Potability Prediction Model

This project aims to predict **whether water is safe for drinking** based on its **physicochemical properties** using various **Machine Learning and Deep Learning** algorithms.  
It demonstrates data preprocessing, visualization, model training, and performance evaluation to explore how water quality parameters relate to potability.

---

## 📘 Project Overview

Unsafe drinking water poses serious health risks worldwide.  
This project uses machine learning to classify water as **potable (1)** or **not potable (0)** using physicochemical test results.

---

## 📊 Dataset

The dataset contains features such as:

- `pH`
- `Hardness`
- `Solids`
- `Chloramines`
- `Sulfate`
- `Conductivity`
- `Organic Carbon`
- `Trihalomethanes`
- `Turbidity`
- `Potability` (Target Variable)

---

## 🧠 Algorithms Used

- Decision Tree Classifier 🌳  
- Random Forest Classifier 🌲  
- XGBoost Classifier ⚡  
- Artificial Neural Network (ANN) 🧩  

---

## ⚙️ Tools & Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf

🧾 Evaluation Metrics

Accuracy

F1-score

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

)

🏆 Results
Model	Accuracy	F1-score	Notes
Decision Tree	65%	Moderate	✅ Best performing
Random Forest	61%	–	–
XGBoost	59%	–	–
ANN	58%	–	–

The Decision Tree Classifier achieved the highest accuracy (65%), showing that tree-based models perform relatively well on this dataset.

📈 Visualizations

Correlation heatmaps

Feature importance plots

Model comparison bar chart

Distribution plots of physicochemical features
