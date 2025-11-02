# symbolic_kd_demo.py
from sklearn.datasets import make_classification 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_classes=2, random_state=42)

teacher = DecisionTreeClassifier(max_depth=3, random_state=42)
teacher.fit(X, y)

teacher_preds = teacher.predict(X)

mean_class1 = np.mean(X[teacher_preds == 1], axis=0)
mean_class0 = np.mean(X[teacher_preds == 0], axis=0)

def symbolic_student(x):
 
    dist1 = np.linalg.norm(x - mean_class1)
    dist0 = np.linalg.norm(x - mean_class0)
    return 1 if dist1 < dist0 else 0

student_preds = np.array([symbolic_student(x) for x in X])
acc = accuracy_score(teacher_preds, student_preds)

print("=== Symbolic Knowledge Distillation Demo ===")
print(f"Teacher → Decision Tree (depth=3)")
print(f"Student → Symbolic IF/ELSE Rules")
print(f"Student imitates Teacher with accuracy: {acc*100:.2f}%")

