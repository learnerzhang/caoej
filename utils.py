from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class SafeSimpleImputer(BaseEstimator, TransformerMixin):
    """安全的SimpleImputer实现，处理众数计算中的边界情况"""
    
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        
    def fit(self, X, y=None):
        if hasattr(X, 'iloc'):  # 如果是DataFrame
            X = X.values
        
        n_features = X.shape[1]
        self.statistics_ = np.zeros(n_features)
        
        for i in range(n_features):
            col_data = X[:, i]
            # 移除NaN值
            valid_data = col_data[~np.isnan(col_data)]
            
            if len(valid_data) == 0:
                # 如果所有值都是NaN，使用fill_value或默认值
                if self.fill_value is not None:
                    self.statistics_[i] = self.fill_value
                else:
                    self.statistics_[i] = 0
                continue
                
            if self.strategy == 'mean':
                self.statistics_[i] = np.mean(valid_data)
            elif self.strategy == 'median':
                self.statistics_[i] = np.median(valid_data)
            elif self.strategy == 'most_frequent':
                # 安全的众数计算
                values, counts = np.unique(valid_data, return_counts=True)
                self.statistics_[i] = values[np.argmax(counts)]
            elif self.strategy == 'constant':
                self.statistics_[i] = self.fill_value if self.fill_value is not None else 0
                
        return self
    
    def transform(self, X):
        if hasattr(X, 'iloc'):  # 如果是DataFrame
            X = X.values.copy()
        else:
            X = X.copy()
            
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if np.any(mask):
                X[mask, i] = self.statistics_[i]
                
        return X
