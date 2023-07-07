import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# define the dataset
X = np.array([[2, 3], [4, 2], [6, 4], [4, 6], [8, 2]])
y = np.array(['Red', 'Red', 'Blue', 'Blue', 'Blue'])

# create a knn classifier with K=3
knn = KNeighborsClassifier(n_neighbors=3)

#  fit the model to the data
knn.fit(X, y)


# Define a new point
new_point = np.array([[5, 5]])


# predict the class of the new point
predicted_class = knn.predict(new_point)

print("Predicted class is: ", predicted_class)


# plot the data set and new points
plt.scatter(X[:0], X[:1], c=y)
plt.scatter(new_point[:0], new_point[:1], c=predicted_class, marker='x', s=100)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KNN Classification')
plt.show()
