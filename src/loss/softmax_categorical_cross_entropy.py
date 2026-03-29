import numpy as np

from activation.softmax import SoftmaxActivation
from loss.categorical_cross_entropy import CategoricalCrossEntropyLoss

class SoftmaxCategoricalCrossEntropyAcitvationLoss():

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape == 2):
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples