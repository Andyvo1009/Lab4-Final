from pyspark.ml.linalg import DenseVector
import numpy as np

class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, features):
        features = np.array(features)
        return ((features - self.mean) / self.std).tolist()

class Transforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, features):
        for transform in self.transforms:
            features = transform(features)
        return features

