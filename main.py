from kit_learn import LinearRegression

model = LinearRegression()

X = [2, 4, 5, 6, 8, 10]

y = [60, 70, 75, 85, 90, 95]

model.fit(X, y)
print(model.predict([7]))