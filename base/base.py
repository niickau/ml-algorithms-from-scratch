# coding:utf-8
import numpy as np

from settings import check_argument


class BaseSL:
    """Generic class for supervised learning algorithms."""
    
    def __init__(self):        
        self.fitted = False
    
    def setup_fit_input(self, X, y):
        """
        Check the validity of the arguments for fitting, provide correct format and prepare the class object.
        
        Args:
        
            X: feature dataset.
            y: target values.
        """
        X = check_argument(X)
        y = check_argument(y)
        
        if X.shape[0] == y.shape[0]:
            if X.shape[0] > 0:
                self.X_train = X
                self.y_train = y
                self.n_samples = X.shape[0]
                self.n_features = X.shape[1]
                self.fitted = True
            else:
                raise ValueError("Train matrices must be non empty.")
        else:
            raise ValueError("Train matrices must have equal number of samples.")
            
    def setup_predict_input(self, X):
        """
        Check the validity of the arguments for predicting, provide correct format 
        and prepare the class object.
        
        Args:
            X: feature dataset.
        """
        if not self.fitted:
            raise ValueError("Estimator must be fitted before predicting.")
            
        X = check_argument(X)
        
        if X.shape[0] > 0:
            if X.shape[1] == self.n_features:
                self.X_test = X
            else:
                raise ValueError("Test and train vesctors must have the same dimensions.")
        else:
            raise ValueError("Test matrix must be non empty.")
