from kit_learn import LinearRegression

model = LinearRegression()

X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] 

y = [4, 7, 10, 13]

model.fit(X, y)
print(model.predict([[2, 3, 4], [5, 6, 7]]))