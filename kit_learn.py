class LinearRegression:
    def __init__(self, alpha = 0.01, nb_iterations=1000) -> None:
        self.alpha = alpha
        self.nb_iterations = nb_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.w = 0
        self.b = 0
        m = len(X)

        for k in range(self.nb_iterations):
            y_hat = [((self.w * X[i]) + self.b) for i in range(m)]

            e = [y_hat[i] - y[i] for i in range(m)]
            
            for i in range(m):
                self.w -= self.alpha * 1/m * e[i]* X[i]
                self.b -= self.alpha * 1/m  * e[i]

            e_square = [e[i]**2 for i in range(m)]

            J = (1/(2*m)) * sum(e_square)

    def predict(self, x):
        return [self.w * x[i] + self.b for i in range(len(x))]