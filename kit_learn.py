class LinearRegression:
    def __init__(self, alpha = 0.001, nb_iterations=1000) -> None:
        self.alpha = alpha
        self.nb_iterations = nb_iterations
        self.w = None
        self.b = None

    def fit(self, X, y) -> None:
        m = len(X)
        n = len(X[0])
        self.w = [0]*m
        self.b = 0

        for k in range(self.nb_iterations):
            y_hat = [sum([((self.w[i] * X[i][l]) + self.b) for l in range(n)]) for i in range(m)]

            e = [y_hat[i] - y[i] for i in range(m)]
            
            for i in range(m):
                    self.w[i] -= self.alpha * 1/m * sum([e[i]* X[i][l] for l in range(n)])

            self.b -= self.alpha * 1/m  * sum(e)

            e_square = [e[i]**2 for i in range(m)]
            J = (1/(2*m)) * sum(e_square)

    def predict(self, x)-> list:
        return [(sum([self.w[l] * x[i][l] for l in range(len(x[0]))]) + self.b) for i in range(len(x))]