import numpy as np

from accuracy.accuracy import Accuracy

class CategoricalAccuracy(Accuracy):

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y