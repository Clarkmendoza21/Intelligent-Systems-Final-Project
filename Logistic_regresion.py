import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset from the CSV file
data = pd.read_csv("datasets/obesity_data.csv")

# Separate features (X) and target variable (y)
X = data.drop("ObesityCategory", axis=1)  # Replace "target_column_name" with the actual column name
y = data["ObesityCategory"]


le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression(max_iter=2000)  # Increase max_iter if needed

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))