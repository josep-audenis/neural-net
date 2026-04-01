import numpy as np
import pickle
import copy

from activation.softmax import SoftmaxActivation
from layer.input import InputLayer
from loss.loss import Loss
from loss.categorical_cross_entropy import CategoricalCrossEntropyLoss
from loss.softmax_categorical_cross_entropy import SoftmaxActivationCategoricalCrossEntropyLoss

class Model:

    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None


    def add_layer(self, layer):
        self.layers.append(layer)


    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy


    def finalize(self):
        self.input_layer = InputLayer()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

            if self.loss is not None:
                self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], SoftmaxActivation) and isinstance(self.loss, CategoricalCrossEntropyLoss):
            self.softmax_classifier_output = SoftmaxActivationCategoricalCrossEntropyLoss()


    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y)
        train_steps = 1
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
        
        for epoch in range(1, epochs+1):
            print(f"epoch: {epoch}")
            
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                
                output = self.forward(batch_X, training=True)
                
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                
                predictions = self.output_layer_activation.predictions(output)
                
                accuracy = self.accuracy.calculate(predictions, batch_y)
                
                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f"Step: {step}\n\tAcc: {accuracy:.2f}\n\tLoss: {loss:.3f} (Data Loss: {data_loss:.3f}, Regularization Loss{regularization_loss:.3f})\n\tLearning Rate: {self.optimizer.current_learning_rate}\n\n")

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            
            print(f"Training\n\tAcc: {epoch_accuracy:.2f}\n\tLoss: {epoch_loss:.3f} (Data Loss: {epoch_data_loss:.3f}, Regularization Loss{epoch_regularization_loss:.3f})\n\tLearning Rate: {self.optimizer.current_learning_rate}\n\n")

            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)


    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1
        
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
            
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f"Validation\n\tAccuracy: {validation_accuracy:.3f}\n\tLoss: {validation_loss:.3f}")


    def predict(self, X, *, batch_size=None):
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X

            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)

        return np.vstack(output)
    

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output
    

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return
        
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        
        return parameters
    

    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    
    def save_parameters(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.get_parameters(), file)

    
    def load_parameters(self, path):
        with open(path, "rb") as file:
            self.set_parameters(pickle.load(file))


    def save(self, path):
        model = copy.deepcopy(self)
        
        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop("output", None)
        model.loss.__dict__.pop("dinputs", None)

        for layer in model.layers:
            for property in ["inputs", "outputs", "dinputs", "dweights", "dbiases"]:    # TODO: Review outputs attribute (output or outputs)
                layer.__dict__.pop(property, None)

        with open(path, "wb") as file:
            pickle.dump(model, file)

    
    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            model = pickle.load(file)

        return model
    

