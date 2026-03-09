import numpy as np

class ReLUActivation:
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)


    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


    def predictions(self, outputs):
        return outputs