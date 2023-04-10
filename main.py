from kit_learn import LinearRegression
import numpy as np

model = LinearRegression()

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array([4, 7, 10, 13])

model.fit(X, y)

X_new = np.array([[2, 3, 4], [5, 6, 7]])
p = model.predict(X_new)

print(p)