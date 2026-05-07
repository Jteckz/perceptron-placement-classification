Perceptron Classification on Placement Dataset

This project demonstrates the implementation of a single-unit Perceptron classifier using scikit-learn for binary classification on a placement dataset.

The objective is to classify whether a student is placed (1) or not placed (0) using:

CGPA
Placement Exam Marks

The project also visualizes the dataset distribution and the decision boundary learned by the perceptron model.

Project Objective

The main goals of this project are:

Understand how a single-layer perceptron works
Train a perceptron classifier on a binary dataset
Visualize decision boundaries
Analyze model performance
Explore the limitations of linear classifiers
Dataset Features

The dataset contains:

Feature	Description
cgpa	Student CGPA
placement_exam_marks	Placement exam score
placed	Target class (0 = Not Placed, 1 = Placed)
Technologies Used
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Mlxtend
Installation

Clone the repository:

git clone https://github.com/your-username/perceptron-placement-classification.git

Move into the project directory:

cd perceptron-placement-classification

Install dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn mlxtend
Project Structure
perceptron-placement-classification/
│
├── placement.csv
├── perceptron_model.ipynb
├── README.md
└── requirements.txt
Implementation
Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
Load Dataset
df = pd.read_csv('placement.csv')
Separate Features and Labels
X = df.iloc[:,0:2]
y = df.iloc[:,-1]
Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Feature scaling standardizes the data to improve model training stability.


Train Perceptron Model
p = Perceptron()

p.fit(X_scaled, y)


b is the bias term
Model Parameters
print("Weights:", p.coef_)
print("Bias:", p.intercept_)

Example output:

Weights: [[1.18585451 0.62757389]]
Bias: [0.]
Accuracy Evaluation
accuracy = p.score(X_scaled, y)

print("Accuracy:", accuracy)

Example output:

Accuracy: 0.508
Dataset Visualization
sns.scatterplot(
    x=X.iloc[:,0],
    y=X.iloc[:,1],
    hue=y
)

plt.title("Dataset Visualization")
plt.show()
Decision Boundary Visualization
plot_decision_regions(
    X_scaled,
    y.values,
    clf=p,
    legend=2
)

plt.show()
Analysis and Findings

The perceptron achieved approximately 50% accuracy, which indicates poor classification performance.

Reason for Low Accuracy

The dataset is not linearly separable.

Observations from the visualization:

Heavy overlap exists between the two classes
No clear straight-line separation is visible
Both classes occupy similar regions in feature space

Since a perceptron can only learn linear boundaries, it struggles to classify overlapping data effectively.

Key Learning Outcomes

This project demonstrates several important machine learning concepts:

Single-layer perceptrons only work well on linearly separable data
Feature scaling is important for stable training
Visualization is critical for understanding model behavior
Model assumptions strongly affect performance
Low accuracy may result from data geometry rather than coding errors
Limitations

The perceptron model has several limitations:

Cannot model nonlinear decision boundaries
Sensitive to overlapping classes
Performs poorly on noisy datasets
Limited expressive power compared to modern ML models
Possible Improvements

Future improvements may include:

Logistic Regression
Support Vector Machines (SVM)
Decision Trees
Random Forests
Neural Networks

Additional features may also improve predictive performance.