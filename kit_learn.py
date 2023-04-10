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
        Fit the linear regression model to the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,)
            The target values.

        Returns
        -------
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
        Predict the output of the linear regression model on new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted target values.

        """
        result = []
        for i in range(x.shape[0]):
            result.append(np.sum(np.dot(self.w, x[i])) + self.b)

        return result
    
    def zscore_normalize_features(self, X):
        """
        Normalize the input features of X using the z-score normalization method.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_norm : ndarray of shape (n_samples, n_features)
            The normalized input data.
        mu : ndarray of shape (n_features,)
            The mean of each feature in X.
        sigma : ndarray of shape (n_features,)
            The standard deviation of each feature in X.

        """
        mu = np.mean(X, axis=0)                 
        sigma = np.std(X, axis=0)          
        X_norm = (X - mu) / sigma      

        return X_norm