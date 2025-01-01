from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
# 1. Load your dataset
# Replace 'your_data.csv' with the actual path to your dataset
import pandas as pd
data = pd.read_csv('datasets/obesity_data.csv')

# 2. Separate features (X) and target variable (y)
X = data.drop('ObesityCategory', axis=1)  # Replace 'target_column' with the actual column name
y = data['ObesityCategory']

le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the value of 'n_neighbors' (k)
knn.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = knn.predict(X_test)

# 6. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Optional: Further evaluation (e.g., classification report, confusion matrix)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))