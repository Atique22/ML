import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

# define the dataset
X = np.array([[2, 3], [4, 2], [6, 4], [4, 6], [4, 5], [7, 5], [5, 7]])
y = np.array(['Red', 'Blue', 'Blue', 'Red', 'Red', 'Red', 'Blue'])

# create a knn classifier with K=3
knn = KNeighborsClassifier(n_neighbors=3)

#  fit the model to the data
knn.fit(X, y)


# Define a new point
new_point = np.array([[5, 5]])


# predict the class of the new point
predicted_class = knn.predict(new_point)

print("Predicted class is: ", predicted_class)


# Calculate accuracy, precision, and recall
y_pred = knn.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, pos_label='Blue')
recall = recall_score(y, y_pred, pos_label='Blue')

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# plot the data set and new points
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(new_point[:, 0], new_point[:, 1],
            c=predicted_class, marker='x', s=100)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KNN Classification')
plt.show()
