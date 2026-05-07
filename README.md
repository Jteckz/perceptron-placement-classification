# Perceptron Classification on Placement Dataset

This project demonstrates the implementation of a single-unit perceptron classifier using Scikit-learn on a binary classification dataset.

The project compares:

1. Perceptron without feature scaling
2. Perceptron with feature standardization

The goal is to understand:
- how perceptrons work,
- how decision boundaries are formed,
- and how feature scaling affects machine learning models.

---

# Dataset Description

The dataset contains:

| Feature | Description |
|---|---|
| cgpa | Student CGPA |
| placement_exam_marks | Placement exam marks |
| placed | Target label (0 = Not Placed, 1 = Placed) |

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Mlxtend

---

# Project Structure

```text
perceptron-placement-classification/
│
├── placement.csv
├── perceptron_scaled.py
├── perceptron_scaled.py
├── README.md
└── requirements.txt