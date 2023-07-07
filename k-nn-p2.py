import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# define the dataset
X = np.array([[70], [85], [50], [65], [80]])
y = np.array(['YES', 'YES', 'NO', 'NO', 'YES'])

# create a knn classifier with K=3
knn = KNeighborsClassifier(n_neighbors=3)

#  fit the model to the data
knn.fit(X, y)


# Define a new point
new_score = np.array([[60]])


# predict the class of the new point
predicted_class = knn.predict(new_score)

print("Predicted class is: ", predicted_class)


# Plot the dataset and the new student
plt.scatter(X, y)
plt.scatter(new_score, predicted_class, marker='x', c='r', s=100)
plt.xlabel('Exam Score')
plt.ylabel('Passed')
plt.title('KNN Classification')
plt.show()
