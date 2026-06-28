# ============================================
# PERCEPTRON WITHOUT FEATURE SCALING
# ============================================

# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron


# ============================================
# Load Dataset
# ============================================
df = pd.read_csv('placement 3.csv')

print(df.head())


# ============================================
# Separate Features and Labels
# ============================================
X = df.iloc[:, 0:2]
y = df.iloc[:, -1]


# ============================================
# Visualize Dataset
# ============================================
sns.scatterplot(
    x=X.iloc[:,0],
    y=X.iloc[:,1],
    hue=y
)

plt.title("Dataset Visualization")
plt.show()


# ============================================
# Create Perceptron Model
# ============================================
p = Perceptron()


# ============================================
# Train Model
# ============================================
p.fit(X, y)


# ============================================
# Model Parameters
# ============================================
print("Learned Weights (coef_): ", p.coef_)
print("Learned Bias (intercept_): ", p.intercept_)


# ============================================
# Accuracy
# ============================================
accuracy = p.score(X, y)

print("The accuracy is ", accuracy)


