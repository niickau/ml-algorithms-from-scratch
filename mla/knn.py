# coding:utf-8
import numpy as np

from base import BaseSL
from settings import KD_Node, check_argument, get_sub_indices


class KD_Tree(BaseSL):
    def __init__(self, k=3, mode='regressor', similarity_metric='euclidean', min_samples_leaf=5):
        """
        Base class for kNN estimator with k-dimensional tree structure.
        
        Args:       
            k: number of nearest neighbors, default 3.
            mode: type of estimator ('classifier' or 'regressor').
            similarity_metric: method of calculating similarity ('random', 'euclidean', 'cosine' or 'correlation').
            min_samples_leaf: minimum number of objects in a leaf of kd-tree, default 5.
        """
        
        self.k = k
        
        if mode in ('classifier', 'regressor'):
            self.mode = mode
        else:
            raise ValueError('Unknown value of "mode".')
            
        if similarity_metric in ('random', 'euclidean', 'cosine', 'correlation'):
            self.similarity_metric = similarity_metric
        else:
            raise ValueError('Unknown value of "similarity_metric".')
            
        if min_samples_leaf >= self.k:
            self.min_samples_leaf = min_samples_leaf
        else:
            raise ValueError('Argument "min_samples_leaf" must be not less then k.')
 
    def fit(self, X_train, y_train):
        super().setup_fit_input(X_train, y_train)
        
        root = self.build_kd_tree(X=self.X_train, curr_depth=0)
        self.kd_tree = root
           
    def predict(self, X_test):     
        super().setup_predict_input(X_test)
        
        preds = []
        for obj in self.X_test:
            ans = self.make_prediction(obj=obj, node=self.kd_tree)
            preds.append(ans)
            
        return np.array(preds)
     
    def build_kd_tree(self, X, curr_depth):
        """
        Translate data from matrix structure to k-dimensional tree structure.
    
        Args:
            X: subset of the original objects at the current_node.
            curr_depth: depth of the current node.
    
        """
        node = KD_Node(curr_depth)
    
        if len(X) > self.min_samples_leaf:
            dimension_to_divide = curr_depth % self.n_features
            median_index = len(X) // 2
        
            X_sort = X[X[:, dimension_to_divide].argsort()]
        
            node.divide_dimension = dimension_to_divide
            node.data = X[median_index]
        
            node.left = self.build_kd_tree(X_sort[:median_index], curr_depth + 1)
            node.right = self.build_kd_tree(X_sort[median_index:], curr_depth + 1)
        else:
            node.data = X
        
        return node

    def make_prediction(self, obj, node):
        """
        Calculate predictions based on k objects in the leaf of kd-tree and the selected metric.
    
        Args:
            obj: object to predict.
            node: current node of kd-tree.
    
        """
        if node.left is None and node.right is None:
            if len(node.data) == self.k:
                sub_X = node.data        
            else:
                if self.similarity_metric == 'random':
                    sample_indices = self.get_indices_on_random(node.data)
                else:
                    sample_indices = self.get_indices_on_metric(obj, node.data)           

                sub_X = []
                for ind in sample_indices:
                    sub_X.append(node.data[ind])
            
            indices = get_sub_indices(np.array(sub_X), self.X_train)
        
            if self.mode == 'regressor':
                accum = 0
                for index in indices:
                    accum += self.y_train[index]
                return accum / len(indices)
        
            elif self.mode == 'classifier':
                targets = [self.y_train[i] for i in indices]
                counts = np.bincount(targets)
                return np.argmax(counts)
    
        else:
            if obj[node.divide_dimension] <= node.data[node.divide_dimension]:
                result = self.make_prediction(obj, node.left)
            else:
                result = self.make_prediction(obj, node.right)
    
        return result            

    def get_indices_on_random(self, data):
        sample_indices = random.sample(range(len(data)), self.k)
        return sample_indices

    def get_indices_on_metric(self, obj, data):
        dict_values = {}
    
        if self.similarity_metric == 'euclidean':
            metric = euclidean
        elif self.similarity_metric == 'cosine':
            metric = cosine
        elif self.similarity_metric == 'correlation':
            metric = correlation

        for i, obj in enumerate(data):
            dict_values[i] = metric(data[i], obj)

        dict_values = OrderedDict(sorted(dict_values.items(), key=lambda t: t[1], reverse=True))
    
        sample_indices = []
        for key in dict_values.keys():
            if len(sample_indices) < self.k:
                sample_indices.append(key)
            else:
                break
    
        return sample_indices
