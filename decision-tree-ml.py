import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Read the dataset from CSV file
data = pd.read_csv('dst_data.csv')

# Step 2: Prepare the dataset
X = data[['Age', 'Gender', 'Income']]
y = data['Purchase']

# Step 3: Perform one-hot encoding on categorical variables
X_encoded = pd.get_dummies(X)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42)

# Step 5: Create and train the decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 8: Make predictions for a new customer
# new_customer = [[27, 1, 0, 0, 1, 0, 1, 0]]
# prediction = model.predict(new_customer)
# print("Prediction for the new customer:", prediction)
