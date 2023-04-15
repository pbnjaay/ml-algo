from kit_learn import LinearRegression
from kit_learn import LogisticRegression
import numpy as np


X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_train = np.array([3, 6, 9])

X_test = np.array([[2, 3, 4], [5, 6, 7], [8,9,10], [11,10,12]])
y_test = np.array([4, 7, 10, 12])

model = LinearRegression(alpha=0.3, iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2_score = model.r2_score(y_pred, y_test)

print('Test value: ', y_test)
print('Prediction test: ', y_pred)
print('Score R²: ', r2_score)

# x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
# y_train = np.array([0, 0, 0, 1, 1, 1])

# x_test = np.array([[0.25, 1.75], [2.5, 0.25], [1.75, 2.25], [1, 1.5], [3, 1.5], [1.5, 1]])
# y_test = np.array([0, 1, 1, 0, 1, 0])

# model = LogisticRegression()

# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# print(y_pred)
