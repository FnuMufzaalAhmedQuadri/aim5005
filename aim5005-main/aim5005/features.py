import numpy as np
from typing import List, Tuple, Union
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. If it can't be cast, raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x

    def fit(self, x: np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum

        # Ensure no division by zero
        diff_max_min[diff_max_min == 0] = 1.0

        # Perform element-wise scaling
        return (x - self.minimum) / diff_max_min
    
    def fit_transform(self, x: list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x

    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        
        # Avoid division by zero by setting zero std to 1
        std = np.where(self.std == 0, 1.0, self.std)

        # Standardization: (x - mean) / std deviation
        return (x - self.mean) / std

    def fit_transform(self, x: list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

class LabelEncoder:
    def __init__(self):
        self.classes_ = None  # Stores the unique classes found during fitting

    def fit(self, y: Union[List, np.ndarray]) -> None:
        """
        Fit the LabelEncoder to the input labels.
        """
        y = np.asarray(y)  # Convert input to numpy array
        self.classes_ = np.unique(y)  # Store unique classes

    def transform(self, y: Union[List, np.ndarray]) -> np.ndarray:
        """
        Transform labels to encoded integers.
        """
        y = np.asarray(y)  # Convert input to numpy array
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted yet.")
        return np.searchsorted(self.classes_, y)  # Map labels to indices

    def fit_transform(self, y: Union[List, np.ndarray]) -> np.ndarray:
        """
        Fit the encoder and transform the labels in one step.
        """
        self.fit(y)
        return self.transform(y)