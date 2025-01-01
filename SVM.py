import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset from CSV file
data = pd.read_csv("datasets/obesity_data.csv")

# Separate features (X) and target variable (y)
X = data.drop("ObesityCategory", axis=1)  # Replace "target_column_name"
y = data["ObesityCategory"]

le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier (you can choose different kernels: 'linear', 'rbf', 'poly', etc.)
svm_model = SVC(kernel='linear')  # Example with linear kernel

# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)