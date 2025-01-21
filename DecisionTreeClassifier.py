import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score,recall_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('datasets/obesity_data.csv')

X = data.drop("ObesityCategory", axis=1)
y = data["ObesityCategory"]

le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1 = f1_score(y_test, y_pred, average=None)
print("F1-Score:", f1)
precision = precision_score(y_test, y_pred, average=None)
print("Precision:", precision)
recall = recall_score(y_test, y_pred, average=None)
print("Recall:", recall)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualization of confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

categories = clf.classes_
metrics_df = pd.DataFrame({
    'Category': categories,
    'F1 Score': f1,
    'Precision': precision,
    'Recall': recall,
    'Accuracy': [accuracy] * len(categories)
})

plt.figure(figsize=(12, 6))
for metric in ['F1 Score', 'Precision', 'Recall', 'Accuracy']:
    plt.plot(metrics_df['Category'], metrics_df[metric], label=metric, marker='o')

plt.title("Metrics by Category")
plt.xlabel("Category")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.show()