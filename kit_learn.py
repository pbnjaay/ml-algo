import numpy as np

class LinearRegression:
    """
    A simple implementation of linear regression using gradient descent.

    Parameters
    ----------
    alpha : float, optional
        The learning rate of the gradient descent algorithm. Default is 0.001.
    nb_iterations : int, optional
        The number of iterations to run the gradient descent algorithm. Default is 1000.

    Attributes
    ----------
    w : ndarray of shape (n_features,)
        The coefficients of the linear regression model.
    b : float
        The intercept of the linear regression model.

    Methods
    -------
    fit(X, y)
        Fit the linear regression model to the training data.
    predict(X)
        Predict the output of the linear regression model on new data.
    zscore_normalize_features(X)
        Normalize the input features of X using the z-score normalization method.

    """
    def __init__(self, alpha = 0.001, nb_iterations=1000) -> None:
        self.alpha = alpha
        self.nb_iterations = nb_iterations
        self.w = None
        self.b = None

    def fit(self, X, y) -> None:
        """
        Fits the LinearRegression model.

        Args:
          X (ndarray (m,n)): input data, m examples, n features
          y (ndarray (m,)): target variable

        Returns:
          None
        """
        m = X.shape[1]
        self.w = np.array([0]*m)
        self.b = 0

        for k in range(self.nb_iterations):
            y_pred = np.dot(X, self.w) + self.b

            residue = y_pred - y

            for i in range(m):
                    self.w[i] -= self.alpha * 1/m * np.sum(np.dot(residue, X))

            self.b -= self.alpha * 1/m  * np.sum(residue)
            
            # J = (1/(2*m)) * np.sum(np.square(residue))
            
            # if k % 100 == 0:
            #     print(k, ' cost: ', J)
                
    def predict(self, x)-> list:
        """
        Predicts the target variable.

        Args:
          x (ndarray (m,n)): input data, m examples, n features

        Returns:
          result (list): predicted target variable for each example
        """
        result = []
        for i in range(x.shape[0]):
            result.append(np.sum(np.dot(self.w, x[i])) + self.b)

        return result
    
    def zscore_normalize_features(self, X):
        """
        computes  X, zcore normalized by column

        Args:
          X (ndarray (m,n))     : input data, m examples, n features

        Returns:
          X_norm (ndarray (m,n)): input normalized by column
          mu (ndarray (n,))     : mean of each feature
          sigma (ndarray (n,))  : standard deviation of each feature
        """
        mu = np.mean(X, axis=0)                 
        sigma = np.std(X, axis=0)          
        X_norm = (X - mu) / sigma      

        return X_norm